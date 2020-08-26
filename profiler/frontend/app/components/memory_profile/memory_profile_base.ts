import {MemoryProfileProtoOrNull} from 'org_xprof/frontend/app/common/interfaces/data_table';

/** Base class of Memory Profile component. */
export class MemoryProfileBase {
  data: MemoryProfileProtoOrNull = null;
  hasMemoryData = false;
  memoryIds: string[] = [];
  selectedMemoryId = '';

  parseData(data: MemoryProfileProtoOrNull) {
    this.data = data;
    if (data && data.memoryIds && data.numHosts) {
      if (data.numHosts > 0 && data.memoryIds.length > 0) {
        this.hasMemoryData = true;
        this.memoryIds = data.memoryIds;
        if (this.selectedMemoryId === '') {
          this.selectedMemoryId = data.memoryIds[0];
        }
      }
    }
  }
}
