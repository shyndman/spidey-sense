interface ImportMetaEnv {
  readonly WXT_ONNX_WASM_FILENAME: string;
  readonly WXT_ONNX_WASM_MODULE_FILENAME: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
