import {Component, EventEmitter, Input, OnChanges, Output, SimpleChanges} from '@angular/core';
import {Node} from 'org_xprof/frontend/app/common/interfaces/op_profile.proto';
import * as utils from 'org_xprof/frontend/app/common/utils/utils';

/** An op table entry view component. */
@Component({
  selector: 'op-table-entry',
  templateUrl: './op_table_entry.ng.html',
  styleUrls: ['./op_table_entry.scss']
})
export class OpTableEntry implements OnChanges {
  /** The depth of node. */
  @Input() level: number = 0;

  /** The main node. */
  @Input() node?: Node;

  /** The selected node. */
  @Input() selectedNode?: Node;

  /** The property to sort by waste time. */
  @Input() byWasted: boolean = false;

  /** The property to show top 90%. */
  @Input() showP90: boolean = false;

  /** The number of children nodes to be shown. */
  @Input() childrenCount: number = 10;

  /** The event when the mouse enter or leave. */
  @Output() hover = new EventEmitter<Node|null>();

  /** The event when the selection is changed. */
  @Output() selected = new EventEmitter<Node>();

  children: Node[] = [];
  expanded: boolean = false;
  barWidth: string = '';
  flameColor: string = '';
  hideFlopsUtilization: boolean = false;
  name: string = '';
  offset: string = '';
  percent: string = '';
  provenance: string = '';
  timeWasted: string = '';
  utilization: string = '';
  numLeftOut: number = 0;

  ngOnChanges(changes: SimpleChanges) {
    if (!this.node) {
      this.children = [];
      return;
    }

    if (this.level === 0) {
      this.expanded = true;
    }
    this.children = this.getChildren();
    this.numLeftOut = this.getLeftOut();
    if (!!this.node && !!this.node.metrics && !!this.node.metrics.time) {
      this.percent = utils.percent(this.node.metrics.time);
      this.barWidth = this.percent;
    } else {
      this.barWidth = '0';
      this.percent = '';
    }
    this.flameColor =
        utils.flameColor(utils.utilization(this.node), 0.7, 1, Math.sqrt);
    this.hideFlopsUtilization = !utils.hasFlops(this.node);
    this.name = (this.node && this.node.name) ? this.node.name : '';
    this.offset = this.level.toString() + 'em';
    this.provenance = (this.node && this.node.xla && this.node.xla.provenance) ?
        this.node.xla.provenance.replace(/^.*(:|\/)/, '') :
        '';
    this.timeWasted = utils.percent(utils.timeWasted(this.node));
    this.utilization = utils.percent(utils.utilization(this.node));
  }

  private get90ChildrenIndex() {
    if (!this.showP90 || !this.node || !this.node.children ||
        this.node.children.length === 0 || !this.node.metrics ||
        !this.node.metrics.time) {
      return this.childrenCount;
    }

    let tot = 0;
    const target90 = this.node.metrics.time * 0.9;
    const targetCount = Math.min(this.childrenCount, this.node.children.length);
    for (let i = 0; i < targetCount; i++) {
      if (tot >= target90) {
        return i;
      }
      const child = this.node.children[i];
      if (child && child.metrics && child.metrics.time) {
        tot += child.metrics.time;
      }
    }
    return this.childrenCount;
  }

  private getChildren(): Node[] {
    if (!this.node || !this.node.children) {
      return [];
    }
    let children: Node[] = [];
    const k = this.get90ChildrenIndex();

    children = this.level ? this.node.children.slice(0, k) :
                            this.node.children.slice();
    if (this.byWasted) {
      children.sort((a, b) => utils.timeWasted(b) - utils.timeWasted(a));
    }

    return children;
  }

  private getLeftOut(): number {
    if (!this.level || !this.node || !this.node.numChildren) return 0;
    return this.node.numChildren -
        Math.min(this.childrenCount, this.children.length);
  }

  toggleExpanded() {
    this.expanded = !this.expanded;
    this.selected.emit(this.node);
  }
}
