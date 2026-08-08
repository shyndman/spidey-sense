interface ImportMetaEnv {
  readonly WXT_ONNX_WASM_FILENAME: string;
  readonly WXT_ONNX_WASM_MODULE_FILENAME: string;
  readonly WXT_MODEL_METADATA_PATH: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

declare module '*?bytes' {
  const bytes: Uint8Array<ArrayBuffer>;
  export default bytes;
}