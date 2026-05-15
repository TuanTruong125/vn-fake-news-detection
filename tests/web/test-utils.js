// Assertion utility: Throws error if condition fails
export function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}


// Report utility: Display test result status (pass/fail) to DOM element
export function report(status, message) {
  const node = document.getElementById("test-result") ?? document.body;
  node.dataset.status = status;
  node.textContent = message;
}
