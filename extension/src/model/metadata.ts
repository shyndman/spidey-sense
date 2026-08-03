import { z } from "zod";

const positiveInteger = z.number().int().positive();
const nonNegativeInteger = z.number().int().nonnegative();
const nonEmptyString = z.string().min(1);
const triple = z
  .tuple([z.number(), z.number(), z.number()])
  .readonly();

const modelLabelSchema = z
  .strictObject({
    index: nonNegativeInteger,
    synset: nonEmptyString,
    label: nonEmptyString,
  })
  .readonly();

const modelSchema = z
  .strictObject({
    id: nonEmptyString,
    filename: nonEmptyString,
    sha256: z.string().regex(/^[0-9a-f]{64}$/),
    sizeBytes: positiveInteger,
    format: z.literal("onnx"),
    opset: positiveInteger,
    sourceUrl: z.url({ protocol: /^https?$/ }),
    sourceRevision: nonEmptyString,
  })
  .readonly();

const inputSchema = z
  .strictObject({
    name: nonEmptyString,
    dataType: z.literal("float32"),
    layout: z.literal("NCHW"),
    shape: z
      .tuple([z.null(), positiveInteger, positiveInteger, positiveInteger])
      .readonly(),
    colorSpace: z.literal("RGB"),
    resizeMode: z.literal("shortest_side"),
    resizeShortestSide: positiveInteger,
    interpolation: z.literal("bilinear"),
    cropMode: z.literal("center"),
    cropWidth: positiveInteger,
    cropHeight: positiveInteger,
    pixelScale: z.number(),
    mean: triple,
    standardDeviation: triple,
  })
  .readonly();

const outputSchema = z
  .strictObject({
    name: nonEmptyString,
    dataType: z.literal("float32"),
    shape: z.tuple([z.null(), positiveInteger]).readonly(),
    activation: z.literal("softmax"),
    labels: z.array(modelLabelSchema).readonly(),
  })
  .readonly();

const classGroupsSchema = z
  .strictObject({
    blocked: z.array(modelLabelSchema).readonly(),
    debug: z.array(modelLabelSchema).readonly(),
  })
  .readonly();

const modelMetadataBaseSchema = z.strictObject({
  schemaVersion: z.literal(1),
  model: modelSchema,
  input: inputSchema,
  output: outputSchema,
  classes: classGroupsSchema,
});

type CandidateMetadata = z.infer<typeof modelMetadataBaseSchema>;
type MetadataRefiner = Parameters<
  typeof modelMetadataBaseSchema.superRefine
>[0];
type RefinementContext = Parameters<MetadataRefiner>[1];

function validateLabels(
  metadata: CandidateMetadata,
  context: RefinementContext,
): void {
  const labels = metadata.output.labels;
  if (labels.length !== metadata.output.shape[1]) {
    context.addIssue({
      code: "custom",
      path: ["output", "labels"],
      message: "label count must equal the output class dimension",
      input: labels,
    });
  }

  const synsets = new Set<string>();
  for (const [position, label] of labels.entries()) {
    if (label.index !== position) {
      context.addIssue({
        code: "custom",
        path: ["output", "labels", position, "index"],
        message: "label indices must be contiguous and zero-based",
        input: label.index,
      });
    }
    if (synsets.has(label.synset)) {
      context.addIssue({
        code: "custom",
        path: ["output", "labels", position, "synset"],
        message: "label synsets must be unique",
        input: label.synset,
      });
    }
    synsets.add(label.synset);
  }
}

function validateInputShape(
  metadata: CandidateMetadata,
  context: RefinementContext,
): void {
  const { input } = metadata;
  if (
    input.shape[1] !== 3 ||
    input.shape[2] !== input.cropHeight ||
    input.shape[3] !== input.cropWidth
  ) {
    context.addIssue({
      code: "custom",
      path: ["input", "shape"],
      message: "input shape must match RGB channels and crop dimensions",
      input: input.shape,
    });
  }
}

function validateClassGroups(
  metadata: CandidateMetadata,
  context: RefinementContext,
): void {
  const claimedIndices = new Map<number, "blocked" | "debug">();
  for (const groupName of ["blocked", "debug"] as const) {
    for (const [position, claimedLabel] of metadata.classes[groupName].entries()) {
      const outputLabel = metadata.output.labels[claimedLabel.index];
      if (
        outputLabel === undefined ||
        outputLabel.index !== claimedLabel.index ||
        outputLabel.synset !== claimedLabel.synset ||
        outputLabel.label !== claimedLabel.label
      ) {
        context.addIssue({
          code: "custom",
          path: ["classes", groupName, position],
          message: "class record must exactly match its indexed output label",
          input: claimedLabel,
        });
      }

      const priorGroup = claimedIndices.get(claimedLabel.index);
      if (priorGroup !== undefined) {
        context.addIssue({
          code: "custom",
          path: ["classes", groupName, position, "index"],
          message:
            priorGroup === groupName
              ? "class group records must be unique"
              : "blocked and debug class groups must not overlap",
          input: claimedLabel.index,
        });
      } else {
        claimedIndices.set(claimedLabel.index, groupName);
      }
    }
  }
}

const validateMetadata: MetadataRefiner = (metadata, context) => {
  validateLabels(metadata, context);
  validateInputShape(metadata, context);
  validateClassGroups(metadata, context);
};

const modelMetadataSchema = modelMetadataBaseSchema
  .superRefine(validateMetadata)
  .readonly();

export type ModelLabel = z.infer<typeof modelLabelSchema>;
export type ModelClassGroup = readonly ModelLabel[];
export type ModelClassGroups = z.infer<typeof classGroupsSchema>;
export type ModelMetadata = z.infer<typeof modelMetadataSchema>;

/**
 * Validates the generated JSON at the extension's untrusted runtime boundary.
 *
 * Callers receive only the deeply readonly, schema-versioned representation used
 * by preprocessing and inference. Zod rejects extra fields before cross-record
 * checks ensure labels and semantic class groups still describe the same output.
 */
export function parseModelMetadata(value: unknown): ModelMetadata {
  return modelMetadataSchema.parse(value);
}

/**
 * Fetches and validates metadata without exposing the requested URL or response
 * body in an error message. The original exception remains available as `cause`
 * for diagnostics, while model bytes and metadata values are never logged.
 */
export async function loadModelMetadata(
  metadataUrl: URL,
  fetcher: typeof fetch = fetch,
): Promise<ModelMetadata> {
  let response: Response;
  try {
    response = await fetcher(metadataUrl);
  } catch (cause: unknown) {
    console.error("Model metadata fetch failed");
    throw new Error("Failed to fetch model metadata", { cause });
  }

  if (!response.ok) {
    console.error(`Model metadata fetch failed with HTTP ${response.status}`);
    throw new Error(`Failed to fetch model metadata: HTTP ${response.status}`);
  }

  let value: unknown;
  try {
    value = await response.json();
  } catch (cause: unknown) {
    console.error("Model metadata JSON decoding failed");
    throw new Error("Failed to decode model metadata JSON", { cause });
  }

  try {
    return parseModelMetadata(value);
  } catch (cause: unknown) {
    console.error("Model metadata validation failed");
    throw new Error("Failed to validate model metadata", { cause });
  }
}

/** Resolves the validated artifact filename beside its injected metadata URL. */
export function resolveModelUrl(
  metadataUrl: URL,
  metadata: ModelMetadata,
): URL {
  return new URL(metadata.model.filename, metadataUrl);
}
