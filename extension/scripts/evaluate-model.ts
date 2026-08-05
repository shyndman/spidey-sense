import { createHash } from "node:crypto";
import { createRequire } from "node:module";
import { mkdir, open, readFile, readdir, rename } from "node:fs/promises";
import { isAbsolute, join, relative, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import initResizeWasm, {
  resize as resizeRgba,
} from "@jsquash/resize/lib/resize/pkg/squoosh_resize.js";
import sharp from "sharp";
import { InferenceSession, Tensor } from "onnxruntime-node";
import { z } from "zod";

import {
  parseModelMetadata,
  type ModelMetadata,
} from "../src/model/metadata";
import {
  transformImageToModelInput,
  type ImagePreprocessingDependencies,
} from "../src/image/preprocessing";
import type { DecodedImage } from "../src/image/decoding";

const SCORE_SCHEMA_VERSION = 1 as const;
const MODEL_METADATA_SUFFIX = ".metadata.json";
const PART_SUFFIX = ".part";
const PROBABILITY_COUNT = 1_000;
const SCORE_CODE_PREFIX = "SCORE_";

const sampleManifestSchema = z
  .strictObject({
    schema_version: z.literal(SCORE_SCHEMA_VERSION),
    sample_id: z.string().min(1),
    source: z.enum(["inaturalist", "coco2017"]),
    source_id: z.string().min(1),
    source_category: z.string().min(1),
    expected_presence: z.enum(["positive", "hard_negative", "broad_negative"]),
    source_url: z.string().min(1),
    license: z.string().min(1),
    image_relative_path: z.string().min(1),
    sha256: z.string().min(1),
    perceptual_hash: z.string().min(1),
    duplicate_group: z.string().min(1),
    split: z.enum(["calibration", "test"]),
    width: z.number().int().positive(),
    height: z.number().int().positive(),
  })
  .readonly();

const scoreRecordSchema = z
  .strictObject({
    schema_version: z.literal(SCORE_SCHEMA_VERSION),
    sample_id: z.string().min(1),
    probabilities: z
      .array(z.number().finite().min(0).max(1))
      .length(PROBABILITY_COUNT),
    blocked_score: z.number().finite().min(0).max(1),
    top_index: z.number().int().min(0).max(PROBABILITY_COUNT - 1),
  })
  .readonly();

type SampleManifest = z.infer<typeof sampleManifestSchema>;
export type ScoreRecord = z.infer<typeof scoreRecordSchema>;
export type ScoreStageSummary = Readonly<{
  schema_version: 1;
  attempted: number;
  completed: number;
  skipped: number;
  failed: number;
}>;

type ScoreProgressEvent =
  | "start"
  | "model_loading"
  | "model_ready"
  | "progress"
  | "complete";

type ScoreProgressPayload = Readonly<{
  readonly stage: "score";
  readonly event: ScoreProgressEvent;
  readonly attempted: number;
  readonly completed: number;
  readonly skipped: number;
  readonly failed: number;
  readonly processed: number;
}>;

export interface ScoringTensor {
  readonly type: string;
  readonly dims: readonly number[];
  readonly data: unknown;
}

export interface ScoringSession {
  readonly inputNames: readonly string[];
  readonly outputNames: readonly string[];
  readonly inputMetadata: readonly SessionValueMetadata[];
  readonly outputMetadata: readonly SessionValueMetadata[];
  run(
    feeds: InferenceSession.FeedsType,
  ): Promise<Readonly<Record<string, ScoringTensor>>>;
  release(): Promise<void>;
}
export interface LoadedScoringModel {
  readonly metadata: ModelMetadata;
  readonly session: ScoringSession;
}

type Decoder = (bytes: Uint8Array) => Promise<DecodedImage>;
type InputTransformer = (
  image: DecodedImage,
  metadata: ModelMetadata,
  dependencies?: ImagePreprocessingDependencies,
) => Promise<Float32Array>;
type SessionFactory = (modelPath: string) => Promise<ScoringSession>;

export interface ScorerDependencies {
  readonly decodeJpeg: Decoder;
  readonly transform: InputTransformer;
  readonly createSession: SessionFactory;
}

export interface ScorerPaths {
  readonly root: string;
  readonly manifests: string;
  readonly images: string;
  readonly annotations: string;
  readonly scores: string;
  readonly errors: string;
  readonly models: string;
}

const nodeRequire = createRequire(import.meta.url);
let resizeWasmInitialization: Promise<void> | undefined;

const nodeImagePreprocessingDependencies: ImagePreprocessingDependencies = {
  resize: async (image, width, height) => {
    resizeWasmInitialization ??= readFile(
      nodeRequire.resolve(
        "@jsquash/resize/lib/resize/pkg/squoosh_resize_bg.wasm",
      ),
    ).then((wasmBytes) => initResizeWasm(wasmBytes).then(() => undefined));
    await resizeWasmInitialization;
    const input = new Uint8Array(
      image.data.buffer,
      image.data.byteOffset,
      image.data.byteLength,
    );
    return {
      width,
      height,
      data: resizeRgba(
        input,
        image.width,
        image.height,
        width,
        height,
        0,
        true,
        false,
      ),
    };
  },
};

const defaultDependencies: ScorerDependencies = {
  decodeJpeg,
  transform: (image, metadata) =>
    transformImageToModelInput(
      image,
      metadata,
      nodeImagePreprocessingDependencies,
    ),
  createSession: async (modelPath) => {
    const session = await InferenceSession.create(modelPath, {
      executionProviders: ["cpu"],
    });
    return {
      inputNames: session.inputNames,
      outputNames: session.outputNames,
      inputMetadata: session.inputMetadata,
      outputMetadata: session.outputMetadata,
      run: (feeds: InferenceSession.FeedsType) => session.run(feeds),
      release: () => session.release(),
    };
  },
};

const failureCodes = {
  INVALID_MANIFEST: `${SCORE_CODE_PREFIX}INVALID_MANIFEST`,
  INVALID_EXISTING_SCORE: `${SCORE_CODE_PREFIX}INVALID_EXISTING_SCORE`,
  MODEL_METADATA: `${SCORE_CODE_PREFIX}MODEL_METADATA`,
  MODEL_LOAD: `${SCORE_CODE_PREFIX}MODEL_LOAD`,
  GRAPH_MISMATCH: `${SCORE_CODE_PREFIX}GRAPH_MISMATCH`,
  IMAGE_PATH: `${SCORE_CODE_PREFIX}IMAGE_PATH`,
  IMAGE_READ: `${SCORE_CODE_PREFIX}IMAGE_READ`,
  IMAGE_DECODE: `${SCORE_CODE_PREFIX}IMAGE_DECODE`,
  IMAGE_PREPROCESS: `${SCORE_CODE_PREFIX}IMAGE_PREPROCESS`,
  INFERENCE: `${SCORE_CODE_PREFIX}INFERENCE`,
  INVALID_OUTPUT: `${SCORE_CODE_PREFIX}INVALID_OUTPUT`,
  SCORE_WRITE: `${SCORE_CODE_PREFIX}SCORE_WRITE`,
} as const;
type FailureCode = (typeof failureCodes)[keyof typeof failureCodes];

interface StageFailure {
  readonly schema_version: 1;
  readonly stage: "score";
  readonly code: FailureCode;
  readonly sample_id: string | null;
}

interface SessionValueMetadata {
  readonly name: string;
  readonly isTensor: boolean;
  readonly type?: string;
  readonly shape?: readonly (number | string)[];
}

interface SessionBoundary {
  readonly inputNames: readonly string[];
  readonly outputNames: readonly string[];
  readonly inputMetadata: readonly SessionValueMetadata[];
  readonly outputMetadata: readonly SessionValueMetadata[];
}

/** Numerically stable softmax over one output row. */
export function softmaxLogits(logits: ArrayLike<number>): Float32Array {
  if (logits.length !== PROBABILITY_COUNT) {
    throw new Error("invalid logits length");
  }
  let maximum = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < logits.length; index += 1) {
    const value = logits[index]!;
    if (!Number.isFinite(value)) {
      throw new Error("non-finite logits");
    }
    maximum = Math.max(maximum, value);
  }

  const probabilities = new Float32Array(logits.length);
  let sum = 0;
  for (let index = 0; index < logits.length; index += 1) {
    const value = Math.exp(logits[index]! - maximum);
    if (!Number.isFinite(value)) {
      throw new Error("non-finite softmax term");
    }
    probabilities[index] = value;
    sum += value;
  }
  if (!Number.isFinite(sum) || sum <= 0) {
    throw new Error("invalid softmax sum");
  }
  for (let index = 0; index < probabilities.length; index += 1) {
    const probability = probabilities[index];
    if (probability === undefined) {
      throw new Error("invalid probability");
    }
    const normalizedProbability = probability / sum;
    if (!Number.isFinite(normalizedProbability)) {
      throw new Error("non-finite probability");
    }
    probabilities[index] = normalizedProbability;
  }
  return probabilities;
}

/** Sums metadata-declared blocked classes without consulting predicted labels. */
export function calculateBlockedScore(
  probabilities: ArrayLike<number>,
  metadata: ModelMetadata,
): number {
  if (probabilities.length !== metadata.output.shape[1]) {
    throw new Error("invalid probability length");
  }
  let total = 0;
  for (const label of metadata.classes.blocked) {
    const probability = probabilities[label.index];
    if (probability === undefined || !Number.isFinite(probability)) {
      throw new Error("invalid blocked probability");
    }
    total += probability;
  }
  if (!Number.isFinite(total) || total < 0) {
    throw new Error("invalid blocked score");
  }
  return Math.min(1, total);
}

/** Validates the single-input/single-output graph against generated metadata. */
export function validateSessionGraph(
  session: SessionBoundary,
  metadata: ModelMetadata,
): void {
  validateBoundary(
    session.inputNames,
    session.inputMetadata,
    metadata.input.name,
    metadata.input.dataType,
    metadata.input.shape,
  );
  validateBoundary(
    session.outputNames,
    session.outputMetadata,
    metadata.output.name,
    metadata.output.dataType,
    metadata.output.shape,
  );
}

function validateBoundary(
  names: readonly string[],
  values: readonly SessionValueMetadata[],
  expectedName: string,
  expectedType: string,
  expectedShape: readonly (number | null)[],
): void {
  const value = values[0];
  if (
    names.length !== 1 ||
    names[0] !== expectedName ||
    values.length !== 1 ||
    value === undefined ||
    value.name !== expectedName ||
    value.isTensor !== true ||
    value.type !== expectedType ||
    value.shape === undefined ||
    !shapesMatch(value.shape, expectedShape)
  ) {
    throw new Error("graph mismatch");
  }
}

function shapesMatch(
  actual: readonly (number | string)[],
  expected: readonly (number | null)[],
): boolean {
  return (
    actual.length === expected.length &&
    actual.every((dimension, index) => {
      const expectedDimension = expected[index];
      if (expectedDimension === null) {
        return typeof dimension === "string";
      }
      return dimension === expectedDimension;
    })
  );
}

/** Converts one JPEG to the extension's unpremultiplied RGBA8 sRGB boundary. */
export async function decodeJpeg(bytes: Uint8Array): Promise<DecodedImage> {
  const source = sharp(bytes, { failOn: "error" });
  const sourceMetadata = await source.metadata();
  if (sourceMetadata.format !== "jpeg") {
    throw new Error("not jpeg");
  }
  const decoded = await source
    .toColorspace("srgb")
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  if (
    decoded.info.channels !== 4 ||
    !Number.isSafeInteger(decoded.info.width) ||
    !Number.isSafeInteger(decoded.info.height) ||
    decoded.info.width <= 0 ||
    decoded.info.height <= 0
  ) {
    throw new Error("invalid decoded jpeg");
  }
  const pixels = new Uint8ClampedArray(decoded.data.length);
  pixels.set(decoded.data);
  return {
    width: decoded.info.width,
    height: decoded.info.height,
    data: pixels,
    channelOrder: "RGBA",
    colorSpace: "srgb",
    alphaMode: "unpremultiplied",
  };
}

function isSafeSampleId(sampleId: string): boolean {
  return /^[A-Za-z0-9._-]+$/.test(sampleId);
}

function safeChild(root: string, child: string): string {
  const absoluteRoot = resolve(root);
  const absoluteChild = resolve(absoluteRoot, child);
  const childRelative = relative(absoluteRoot, absoluteChild);
  if (
    isAbsolute(childRelative) ||
    childRelative === "" ||
    childRelative === ".." ||
    childRelative.startsWith("../")
  ) {
    throw new Error("path outside data root");
  }
  return absoluteChild;
}

function resolveManifestImagePath(
  paths: ScorerPaths,
  imageRelativePath: string,
): string {
  const parts = imageRelativePath.split("/");
  if (
    parts.length < 2 ||
    parts[0] !== "images" ||
    parts.some((part) => part.length === 0 || part === "." || part === "..")
  ) {
    throw new Error("invalid image path");
  }
  const imagePath = safeChild(paths.root, imageRelativePath);
  const imageRelative = relative(resolve(paths.images), imagePath);
  if (
    imageRelative === "" ||
    imageRelative === ".." ||
    imageRelative.startsWith("../")
  ) {
    throw new Error("image path outside image directory");
  }
  return imagePath;
}


function sampleOutputPath(paths: ScorerPaths, sampleId: string): string {
  if (!isSafeSampleId(sampleId)) {
    throw new Error("invalid sample id");
  }
  return join(paths.scores, `${sampleId}.json`);
}

function sampleErrorPath(paths: ScorerPaths, sampleId: string | undefined): string {
  const token =
    sampleId !== undefined && isSafeSampleId(sampleId)
      ? sampleId
      : createHash("sha256").update("manifest").digest("hex");
  return join(paths.errors, `score-${token}.json`);
}

function stageFailure(code: FailureCode, sampleId?: string): StageFailure {
  return {
    schema_version: SCORE_SCHEMA_VERSION,
    stage: "score",
    code,
    sample_id: sampleId ?? null,
  };
}

async function writeJsonAtomically(path: string, value: unknown): Promise<void> {
  await mkdir(resolve(path, ".."), { recursive: true });
  const partPath = `${path}${PART_SUFFIX}`;
  const handle = await open(partPath, "w", 0o600);
  try {
    await handle.writeFile(JSON.stringify(value));
    await handle.sync();
  } finally {
    await handle.close();
  }
  await rename(partPath, path);
}

async function readJson(path: string): Promise<unknown> {
  return JSON.parse(await readFile(path, "utf8"));
}

function parseScore(value: unknown, sampleId: string): ScoreRecord | undefined {
  const parsed = scoreRecordSchema.safeParse(value);
  if (!parsed.success || parsed.data.sample_id !== sampleId) {
    return undefined;
  }
  return parsed.data;
}

async function existingScore(path: string, sampleId: string): Promise<boolean> {
  try {
    return parseScore(await readJson(path), sampleId) !== undefined;
  } catch {
    return false;
  }
}

async function listManifestFiles(paths: ScorerPaths): Promise<string[]> {
  const entries = await readdir(paths.manifests, { withFileTypes: true });
  return entries
    .filter(
      (entry) => entry.isFile() && entry.name.endsWith(".json") && !entry.name.endsWith(PART_SUFFIX),
    )
    .map((entry) => join(paths.manifests, entry.name))
    .sort();
}

async function writeFailure(
  paths: ScorerPaths,
  code: FailureCode,
  sampleId?: string,
): Promise<void> {
  try {
    await writeJsonAtomically(sampleErrorPath(paths, sampleId), stageFailure(code, sampleId));
  } catch {
    // Failure records are best-effort and must never expose the original cause.
  }
}

async function loadModel(
  paths: ScorerPaths,
  dependencies: ScorerDependencies,
): Promise<LoadedScoringModel> {
  const modelEntries = await readdir(paths.models, { withFileTypes: true });
  const metadataNames = modelEntries
    .filter((entry) => entry.isFile() && entry.name.endsWith(MODEL_METADATA_SUFFIX))
    .map((entry) => entry.name)
    .sort();
  if (metadataNames.length !== 1) {
    throw new Error("model metadata unavailable");
  }
  const metadataPath = join(paths.models, metadataNames[0]!);
  const metadata = parseModelMetadata(await readJson(metadataPath));
  const modelPath = safeChild(paths.models, metadata.model.filename);
  const session = await dependencies.createSession(modelPath);
  try {
    validateSessionGraph(session, metadata);
  } catch (cause: unknown) {
    await session.release().catch(() => undefined);
    throw cause;
  }
  return { metadata, session };
}

async function runTransformQuietly(
  transform: InputTransformer,
  image: DecodedImage,
  metadata: ModelMetadata,
): Promise<Float32Array> {
  const debug = console.debug;
  const error = console.error;
  console.debug = () => undefined;
  console.error = () => undefined;
  try {
    return await transform(image, metadata);
  } finally {
    console.debug = debug;
    console.error = error;
  }
}

async function scoreOne(
  paths: ScorerPaths,
  manifest: SampleManifest,
  metadata: ModelMetadata,
  session: ScoringSession,
  dependencies: ScorerDependencies,
): Promise<ScoreRecord> {
  const imagePath = resolveManifestImagePath(paths, manifest.image_relative_path);
  const bytes = await readFile(imagePath);
  const image = await dependencies.decodeJpeg(bytes);
  const input = await runTransformQuietly(dependencies.transform, image, metadata);
  const expectedInputElements =
    metadata.input.shape[1] * metadata.input.shape[2] * metadata.input.shape[3];
  if (input.length !== expectedInputElements || !input.every(Number.isFinite)) {
    throw new Error("invalid model input");
  }
  const inputTensor = new Tensor(
    "float32",
    input,
    [1, metadata.input.shape[1], metadata.input.shape[2], metadata.input.shape[3]],
  );
  const outputMap = await session.run({ [metadata.input.name]: inputTensor });
  const output = outputMap[metadata.output.name];
  if (
    output === undefined ||
    output.type !== "float32" ||
    output.dims.length !== 2 ||
    output.dims[0] !== 1 ||
    output.dims[1] !== PROBABILITY_COUNT ||
    !(output.data instanceof Float32Array)
  ) {
    throw new Error("invalid model output");
  }
  const probabilities = softmaxLogits(output.data);
  let topIndex = 0;
  for (let index = 1; index < probabilities.length; index += 1) {
    if (probabilities[index]! > probabilities[topIndex]!) {
      topIndex = index;
    }
  }
  const blockedScore = calculateBlockedScore(probabilities, metadata);
  return {
    schema_version: SCORE_SCHEMA_VERSION,
    sample_id: manifest.sample_id,
    probabilities: [...probabilities],
    blocked_score: blockedScore,
    top_index: topIndex,
  };
}

function emptySummary(): ScoreStageSummary {
  return {
    schema_version: SCORE_SCHEMA_VERSION,
    attempted: 0,
    completed: 0,
    skipped: 0,
    failed: 0,
  };
}

function emitProgress(
  event: ScoreProgressEvent,
  summary: ScoreStageSummary,
  processed: number,
): void {
  const payload: ScoreProgressPayload = {
    stage: "score",
    event,
    attempted: summary.attempted,
    completed: summary.completed,
    skipped: summary.skipped,
    failed: summary.failed,
    processed,
  };
  process.stderr.write(`${JSON.stringify(payload)}\n`);
}

function completeSummary(
  summary: ScoreStageSummary,
  processed: number,
): ScoreStageSummary {
  emitProgress("complete", summary, processed);
  return summary;
}

/** Scores all manifests, retaining only aggregate counters on stdout via the CLI. */
export async function score(
  paths: ScorerPaths,
  dependencies: ScorerDependencies = defaultDependencies,
): Promise<ScoreStageSummary> {
  let summary = emptySummary();
  let processed = 0;
  emitProgress("start", summary, processed);
  const manifestFiles = await listManifestFiles(paths);
  const manifests: Array<{
    readonly path: string;
    readonly value?: SampleManifest;
    readonly code?: FailureCode;
  }> = [];
  for (const manifestPath of manifestFiles) {
    try {
      const parsed = sampleManifestSchema.parse(await readJson(manifestPath));
      if (!isSafeSampleId(parsed.sample_id)) {
        manifests.push({ path: manifestPath, code: failureCodes.INVALID_MANIFEST });
      } else {
        manifests.push({ path: manifestPath, value: parsed });
      }
    } catch {
      manifests.push({ path: manifestPath, code: failureCodes.INVALID_MANIFEST });
    }
  }

  const pending: SampleManifest[] = [];
  for (const entry of manifests) {
    if (entry.code !== undefined) {
      summary = {
        ...summary,
        attempted: summary.attempted + 1,
        failed: summary.failed + 1,
      };
      await writeFailure(paths, entry.code);
      continue;
    }
    const manifest = entry.value;
    if (manifest === undefined) {
      continue;
    }
    summary = { ...summary, attempted: summary.attempted + 1 };
    if (await existingScore(sampleOutputPath(paths, manifest.sample_id), manifest.sample_id)) {
      summary = { ...summary, skipped: summary.skipped + 1 };
    } else {
      pending.push(manifest);
    }
  }

  const markProcessed = (): void => {
    processed += 1;
    if (processed % 25 === 0) {
      emitProgress("progress", summary, processed);
    }
  };
  if (pending.length === 0) {
    return completeSummary(summary, processed);
  }
  emitProgress("model_loading", summary, processed);

  let model: LoadedScoringModel;
  try {
    model = await loadModel(paths, dependencies);
  } catch {
    for (const manifest of pending) {
      await writeFailure(paths, failureCodes.MODEL_LOAD, manifest.sample_id);
      summary = { ...summary, failed: summary.failed + 1 };
      markProcessed();
    }
    return completeSummary(summary, processed);
  }
  emitProgress("model_ready", summary, processed);

  try {
    for (const manifest of pending) {
      const outputPath = sampleOutputPath(paths, manifest.sample_id);
      try {
        const record = await scoreOne(
          paths,
          manifest,
          model.metadata,
          model.session,
          dependencies,
        );
        await writeJsonAtomically(outputPath, record);
        summary = { ...summary, completed: summary.completed + 1 };
      } catch {
        await writeFailure(paths, failureCodes.INFERENCE, manifest.sample_id);
        summary = { ...summary, failed: summary.failed + 1 };
      }
      markProcessed();
    }
  } finally {
    await model.session.release().catch(() => undefined);
  }
  return completeSummary(summary, processed);
}

function parseDataDir(argv: readonly string[]): string {
  let dataDir: string | undefined;
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index]!;
    if (argument === "--data-dir") {
      dataDir = argv[index + 1];
      index += 1;
    } else if (argument.startsWith("--data-dir=")) {
      dataDir = argument.slice("--data-dir=".length);
    } else {
      throw new Error("invalid arguments");
    }
  }
  return dataDir === undefined || dataDir.length === 0 ? "/data" : resolve(dataDir);
}

function scorerPaths(root: string): ScorerPaths {
  return {
    root,
    manifests: join(root, "manifests"),
    images: join(root, "images"),
    annotations: join(root, "annotations"),
    scores: join(root, "scores"),
    errors: join(root, "errors"),
    models: join(root, "models"),
  };
}


async function runCli(argv: readonly string[]): Promise<ScoreStageSummary> {
  const paths = scorerPaths(parseDataDir(argv));
  return score(paths);
}

const invokedScript = process.argv[1];
if (
  invokedScript !== undefined &&
  import.meta.url === pathToFileURL(resolve(invokedScript)).href
) {
  void runCli(process.argv.slice(2))
    .then((summary) => {
      console.log(JSON.stringify(summary));
      if (summary.failed > 0) {
        process.exitCode = 1;
      }
    })
    .catch(() => {
      const summary: ScoreStageSummary = {
        schema_version: SCORE_SCHEMA_VERSION,
        attempted: 0,
        completed: 0,
        skipped: 0,
        failed: 1,
      };
      emitProgress("complete", summary, 0);
      console.log(JSON.stringify(summary));
      process.exitCode = 1;
    });
}
