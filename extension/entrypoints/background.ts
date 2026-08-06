import { registerImagePassThrough } from '../src/network/image-pass-through';
import { configureOnnxRuntime } from '../src/model/runtime-environment';

export default defineBackground(() => {
  configureOnnxRuntime();
  registerImagePassThrough();
  console.info('spidey-sense background runtime started');
});
