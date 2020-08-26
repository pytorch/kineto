import * as proto from 'org_xprof/frontend/app/common/interfaces/xla_data.proto';
import {toNumber} from 'org_xprof/frontend/app/common/utils/utils';

/** The Dimension interface. */
interface Dimension {
  size?: number;
  alignment?: number;
}

/**
 * A layout describes how the array is placed in (1D) memory space. This
 * includes the minor-to-major ordering of dimensions within a shape, as well
 * as size, alignment and semantics.
 * @final
 */
export class Layout {
  dimensions: Dimension[];
  format: string;
  maxSparseElements: number;
  minorToMajor: number[];

  constructor(
      layout?: proto.LayoutProto, dimensions: number[] = [],
      elementType: string = '') {
    layout = layout || {};
    this.format = layout.format || '';
    this.minorToMajor = [];
    this.dimensions = [];
    this.maxSparseElements = 0;
    if (layout.minorToMajor) {
      this.minorToMajor = layout.minorToMajor.map(item => Number(item));
      this.dimensions =
          this.analyzeLayout(dimensions, this.minorToMajor, elementType);
    }
  }

  private analyzeLayout(
      dimensions: number[], minorToMajor: number[],
      elementType: string): Dimension[] {
    const result = [];
    for (let index of minorToMajor) {
      const size = dimensions[index];
      let alignment = 0;
      if (result.length === 0) {
        alignment = 128;
      } else if (result.length === 1) {
        if (size <= 2) {
          if (elementType === 'BF16') {
            alignment = 4;
          } else {
            alignment = 2;
          }
        } else if (size <= 4) {
          alignment = 4;
        } else {
          alignment = 8;
        }
      }
      result.push({'size': size, 'alignment': alignment});
    }
    return result;
  }

  /**
   * Returns a human-readable string that represents the given layout.
   * @return {string}
   */
  humanLayoutString(): string {
    if (this.format === 'SPARSE') {
      return 'sparse{' + this.maxSparseElements.toString() + '}';
    } else if (this.format === 'DENSE') {
      return '{' + this.minorToMajor.join() + '}';
    }
    return '';
  }
}
