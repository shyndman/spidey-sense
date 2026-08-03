import { configureOnnxRuntime } from '../src/model/runtime-environment';

export default defineBackground(() => {
  configureOnnxRuntime();
  console.info('spidey-sense background runtime started');
});
