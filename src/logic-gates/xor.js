import generic from './generic';

const inputs = [[0, 0], [0, 1], [1, 0], [1, 1],[0, 0], [0, 1], [1, 0], [1, 1],[0, 0], [0, 1], [1, 0], [1, 1]];
const outputs = [[0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0]];

const testInputs = [[0, 0], [0, 1], [1, 0], [1, 1]];

export default generic('XOR', inputs, outputs, testInputs);
