import assert from "node:assert/strict";
import { readdir } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { By, until } from "selenium-webdriver";
import * as firefox from "selenium-webdriver/firefox.js";

const FIREFOX_BINARY = "/usr/bin/firefox-devedition";
const EXTENSION_OUTPUT_DIRECTORY = fileURLToPath(
  new URL("../.output/", import.meta.url),
);
const SMOKE_PAGE_PATH = "/model-smoke.html";
const SMOKE_RESULT_ID = "smoke-result";
const SMOKE_TIMEOUT_MILLISECONDS = 120_000;

interface ExtensionUuidMap {
  readonly [extensionId: string]: string;
}

async function findProductionXpi(): Promise<string> {
  const xpiFiles = (await readdir(EXTENSION_OUTPUT_DIRECTORY, {
    withFileTypes: true,
  })).filter((entry) => entry.isFile() && entry.name.endsWith(".xpi"));
  const [xpiFile, additionalXpiFile] = xpiFiles;
  if (xpiFile === undefined || additionalXpiFile !== undefined) {
    throw new Error("Expected exactly one production XPI artifact");
  }
  return join(EXTENSION_OUTPUT_DIRECTORY, xpiFile.name);
}

function parseExtensionUuidMap(value: string): ExtensionUuidMap {
  const parsed: unknown = JSON.parse(value);
  if (
    typeof parsed !== "object" ||
    parsed === null ||
    Array.isArray(parsed) ||
    Object.values(parsed).some((uuid) => typeof uuid !== "string")
  ) {
    throw new Error("Firefox returned an invalid extension UUID map");
  }
  return parsed as ExtensionUuidMap;
}

async function setFirefoxOffline(driver: firefox.Driver): Promise<void> {
  await driver.setContext(firefox.Context.CHROME);
  try {
    const offline = await driver.executeScript<boolean>(
      "Services.io.offline = true; return Services.io.offline;",
    );
    assert.equal(offline, true, "Firefox did not enter offline mode");
  } finally {
    await driver.setContext(firefox.Context.CONTENT);
  }
}

async function resolveExtensionUuid(
  driver: firefox.Driver,
  extensionId: string,
): Promise<string> {
  await driver.setContext(firefox.Context.CHROME);
  try {
    //! HACK: GeckoDriver's installAddon command returns the manifest extension
    //! ID, while Firefox extension pages use a private per-profile UUID in their
    //! moz-extension URL. WebDriver exposes no public UUID lookup. Reading the
    //! browser-owned mapping from privileged chrome context avoids guessing the
    //! UUID or weakening the extension manifest with a web-accessible test page.
    const serializedMapping = await driver.executeScript<string>(
      'return Services.prefs.getStringPref("extensions.webextensions.uuids");',
    );
    const extensionUuid = parseExtensionUuidMap(serializedMapping)[extensionId];
    if (extensionUuid === undefined) {
      throw new Error("Installed extension has no Firefox internal UUID");
    }
    return extensionUuid;
  } finally {
    await driver.setContext(firefox.Context.CONTENT);
  }
}

/**
 * Installs the exact production XPI into the workstation's Firefox Developer
 * Edition, disables browser networking before installation, and opens the
 * package's otherwise-unlisted smoke page. The page drives one real model
 * inference plus native decoding of harmless, in-memory proxy bytes, publishing
 * only a stable pass/fail marker. WebDriver never captures screenshots or
 * handles image input or artifacts.
 */
async function runFirefoxExtensionSmoke(): Promise<void> {
  const xpiPath = await findProductionXpi();
  //! HACK: Firefox 138+ denies WebDriver's privileged chrome context unless
  //! GeckoDriver starts it with explicit system access. The harness uses that
  //! context only to force offline mode and read Firefox's private extension-
  //! UUID mapping; normal page automation remains in unprivileged content.
  const options = new firefox.Options()
    .setBinary(FIREFOX_BINARY)
    .addArguments("-headless");
  const service = new firefox.ServiceBuilder()
    .addArguments("--allow-system-access")
    .build();
  const driver = firefox.Driver.createSession(options, service);

  try {
    await setFirefoxOffline(driver);
    const extensionId = await driver.installAddon(xpiPath, true);
    const extensionUuid = await resolveExtensionUuid(driver, extensionId);
    await driver.get(
      new URL(SMOKE_PAGE_PATH, `moz-extension://${extensionUuid}`).href,
    );

    const result = await driver.wait(
      until.elementLocated(By.id(SMOKE_RESULT_ID)),
      SMOKE_TIMEOUT_MILLISECONDS,
    );
    const status = await driver.wait(async () => {
      const currentStatus = await result.getAttribute("data-status");
      return currentStatus === "passed" || currentStatus === "failed"
        ? currentStatus
        : false;
    }, SMOKE_TIMEOUT_MILLISECONDS);

    assert.equal(status, "passed", "Firefox extension smoke test failed");
    console.info("Firefox extension smoke test passed");
  } finally {
    await driver.quit();
  }
}

void runFirefoxExtensionSmoke().catch((cause: unknown) => {
  console.error("Firefox extension smoke test failed", cause);
  process.exitCode = 1;
});
