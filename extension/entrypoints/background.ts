import { registerImageReplacement } from '../src/network/image-replacement';
import { configureOnnxRuntime } from '../src/model/runtime-environment';

export default defineBackground(() => {
  configureOnnxRuntime();
  registerImageReplacement();
  console.info('spidey-sense background runtime started');
});
