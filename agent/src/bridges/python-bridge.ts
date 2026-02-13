import { PythonSubprocessBridge } from './python-subprocess.js';
import { PythonHttpBridge } from './python-http.js';

export type PythonBridge = PythonSubprocessBridge | PythonHttpBridge;

export { PythonSubprocessBridge, PythonHttpBridge };
export * from './python-subprocess.js';
