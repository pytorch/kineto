import * as proto from 'org_xprof/frontend/app/common/interfaces/xla_data.proto';
import * as utils from 'org_xprof/frontend/app/common/utils/utils';
import {Layout} from './layout';

/**
 * A shape describes the number of dimensions in the array, the size of each
 * dimension, and the primitive component type. Tuples are a special case in
 * that they have rank zero and have tuple_shapes defined.
 * @final
 */
export class Shape {
  dimensions: number[];
  elementType: string;
  layout?: Layout;
  tupleShapes: Shape[];

  constructor(shape?: proto.ShapeProto) {
    shape = shape || {};
    this.elementType = shape.elementType || '';
    this.dimensions = [];
    this.tupleShapes = [];
    if (shape.dimensions) {
      this.dimensions = shape.dimensions.map(item => Number(item));
    }
    if (shape.tupleShapes) {
      this.tupleShapes = shape.tupleShapes.map(item => new Shape(item));
    }
    if (shape.layout) {
      this.layout = new Layout(
          shape.layout, this.dimensions,
          shape.elementType as proto.PrimitiveType);
    }
  }

  /**
   * Returns a human-readable string that represents the given shape, with
   * layout. e.g. "f32[42x12] {0, 1}"
   * @return {string}
   */
  humanStringWithLayout(): string {
    if (this.elementType === 'TUPLE') {
      return `(${
          this.tupleShapes.map(shape => shape.humanStringWithLayout())
              .join(', ')})`;
    }
    const result = this.elementType + '[' + this.dimensions.join() + ']';
    if (this.elementType !== 'OPAQUE' && this.elementType !== 'TOKEN' &&
        this.dimensions.length > 0 && this.layout) {
      return result + this.layout.humanLayoutString();
    }
    return result;
  }

  /**
   * Resolve the right shape from the shapeIndex.
   * @param {!Array<number>} shapeIndex
   * @return {!Shape}
   */
  resolveShapeIndex(shapeIndex: number[]): Shape {
    return shapeIndex.reduce(
        (shape: Shape, item) => shape.tupleShapes[item], this);
  }

  /**
   * Returns the size of shape with out padding.
   * @return {number}
   */
  unpaddedHeapSizeBytes(): number {
    const INT64_BYTES = 8;
    if (this.elementType === 'TOKEN') {
      return 0;
    }
    if (this.elementType === 'TUPLE') {
      return INT64_BYTES * this.tupleShapes.length;
    }
    let byteSize = 0;
    if (!this.layout || this.layout.format === 'DENSE') {
      const allocatedElementCount =
          this.dimensions.reduce((count, item) => count * item, 1);
      byteSize += allocatedElementCount *
          utils.byteSizeOfPrimitiveType(this.elementType);
    } else if (this.layout.format === 'SPARSE') {
      const maxElements = this.layout.maxSparseElements;
      byteSize = maxElements * utils.byteSizeOfPrimitiveType(this.elementType);
      byteSize += maxElements * this.dimensions.length * INT64_BYTES;
    }
    return byteSize;
  }
}
