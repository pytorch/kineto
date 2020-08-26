import {Node} from 'org_xprof/frontend/app/common/interfaces/op_profile.proto';
import {ProfileOrNull} from 'org_xprof/frontend/app/common/interfaces/data_table';

import {OpProfileData} from './op_profile_data';

/** Base class of Op Profile component. */
export class OpProfileBase {
  profile: ProfileOrNull = null;
  rootNode?: Node;
  data = new OpProfileData();
  hasTwoProfiles: boolean = false;
  isByCategory: boolean = false;
  byWasted: boolean = false;
  showP90: boolean = false;
  childrenCount: number = 10;
  deviceType: string = 'TPU';

  private hasMultipleProfiles(): boolean {
    return !!this.profile && !!this.profile.byCategory &&
        !!this.profile.byProgram;
  }

  private updateRoot() {
    if (!this.profile) {
      this.rootNode = undefined;
      return;
    }
    if (!this.hasTwoProfiles) {
      this.rootNode = this.profile.byCategory || this.profile.byProgram;
    } else {
      this.rootNode =
          this.isByCategory ? this.profile.byCategory : this.profile.byProgram;
    }
    this.deviceType = this.profile.deviceType || 'TPU';
  }

  parseData(data: ProfileOrNull) {
    this.profile = data;
    this.hasTwoProfiles = this.hasMultipleProfiles();
    this.isByCategory = false;
    this.childrenCount = 10;
    this.updateRoot();
    this.data.update(this.rootNode);
  }

  updateChildrenCount(value: number) {
    const rounded = Math.round(value / 10) * 10;

    this.childrenCount = Math.max(Math.min(rounded, 100), 10);
  }

  updateToggle() {
    this.isByCategory = !this.isByCategory;
    this.updateRoot();
  }

  updateByWasted() {
    this.byWasted = !this.byWasted;
  }

  updateShowP90() {
    this.showP90 = !this.showP90;
  }
}
