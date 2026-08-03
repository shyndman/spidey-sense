import { runModelInference } from "../../src/model/inference";
import { ModelSessionManager } from "../../src/model/session";

const PROBABILITY_SUM_TOLERANCE = 1e-5;

const SmokeStatus = {
  PASSED: "passed",
  FAILED: "failed",
} as const;

type SmokeStatus = (typeof SmokeStatus)[keyof typeof SmokeStatus];

/**
 * Exercises the exact packaged metadata, model, and ONNX Runtime assets from a
 * real Firefox extension context. The numeric input is an all-zero synthetic
 * tensor—not an encoded, decoded, or preprocessed image—so this page can prove
 * runtime initialization and inference without acquiring or persisting imagery.
 * The page is unlisted and has no manifest or product UI route; WebDriver opens
 * its internal extension URL directly during the explicit smoke command.
 */
async function runFirefoxModelSmoke(): Promise<void> {
  const metadataUrl = new URL(
    import.meta.env.WXT_MODEL_METADATA_PATH,
    globalThis.location.href,
  );
  const manager = new ModelSessionManager(metadataUrl);
  const { metadata, session } = await manager.initialize();
  const [, channels, height, width] = metadata.input.shape;
  const syntheticInput = new Float32Array(channels * height * width);
  const result = await runModelInference(session, metadata, syntheticInput);

  if (result.probabilities.length !== metadata.output.shape[1]) {
    throw new Error("Smoke output length does not match generated metadata");
  }

  let probabilitySum = 0;
  for (const probability of result.probabilities) {
    if (!Number.isFinite(probability)) {
      throw new Error("Smoke output contains a non-finite probability");
    }
    probabilitySum += probability;
  }
  if (Math.abs(probabilitySum - 1) > PROBABILITY_SUM_TOLERANCE) {
    throw new Error("Smoke output probabilities are not normalized");
  }
}

function publishStatus(status: SmokeStatus): void {
  const result = document.querySelector<HTMLElement>("#smoke-result");
  if (result === null) {
    throw new Error("Smoke result element is missing");
  }
  result.dataset.status = status;
  result.textContent = status;
}

void runFirefoxModelSmoke().then(
  () => publishStatus(SmokeStatus.PASSED),
  () => {
    console.error("Firefox model runtime smoke test failed");
    publishStatus(SmokeStatus.FAILED);
  },
);
