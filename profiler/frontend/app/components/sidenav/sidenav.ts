import {Component, EventEmitter, Input, Output} from '@angular/core';
import {Store} from '@ngrx/store';
import {DEFAULT_HOST} from 'org_xprof/frontend/app/common/constants/constants';
import {NavigationEvent} from 'org_xprof/frontend/app/common/interfaces/navigation_event';
import {Tool} from 'org_xprof/frontend/app/common/interfaces/tool';
import {DataService} from 'org_xprof/frontend/app/services/data_service/data_service';
import {setLoadingStateAction} from 'org_xprof/frontend/app/store/actions';
import {getCurrentTool} from 'org_xprof/frontend/app/store/selectors';

/** A side navigation component. */
@Component({
  selector: 'sidenav',
  templateUrl: './sidenav.ng.html',
  styleUrls: ['./sidenav.scss']
})
export class SideNav {
  /** The tool datasets. */
  @Input()
  set datasets(tools: Tool[]|null) {
    if (tools && tools.length > 0) {
      this.tools = tools;
      this.runs = tools.map(tool => tool.name);
      this.selectedRun = tools[0].name;
      this.updateTags();
    }
  }

  /** Navigation Update Event */
  @Output() update = new EventEmitter<NavigationEvent>();

  private tools: Tool[] = [];

  runs: string[] = [];
  tags: string[] = [];
  hosts: string[] = [];
  selectedRun = '';
  selectedTag = '';
  selectedHost = '';

  constructor(
      private readonly dataService: DataService,
      private readonly store: Store<{}>) {
    store.select(getCurrentTool).subscribe((currentTool: string) => {
      this.updateTags(currentTool);
    });
  }

  getDisplayTagName(tag: string): string {
    return (tag && tag.length &&
            (tag[tag.length - 1] === '@' || tag[tag.length - 1] === '#' ||
             tag[tag.length - 1] === '^')) ?
        tag.slice(0, -1) :
        tag || '';
  }

  updateTags(targetTag: string = '') {
    this.store.dispatch(setLoadingStateAction({
      loadingState: {
        loading: true,
        message: 'Loading data',
      }
    }));

    const tool = this.tools.find(tool => tool.name === this.selectedRun);
    if (tool && tool.activeTools && tool.activeTools.length > 0) {
      this.tags = tool.activeTools;
      this.selectedTag =
          this.tags.find(
              tag => tag === targetTag || tag === targetTag + '@' ||
                  tag === targetTag + '#' || tag === targetTag + '^') ||
          this.tags[0];
      this.updateHosts();
    } else {
      this.tags = [];
      this.selectedTag = '';
      this.hosts = [];
      this.selectedHost = '';
      this.emitUpdateEvent();
    }
  }

  updateHosts() {
    this.store.dispatch(setLoadingStateAction({
      loadingState: {
        loading: true,
        message: 'Loading data',
      }
    }));

    const run = this.selectedRun;
    const tag = this.selectedTag;
    this.hosts = [];
    this.selectedHost = '';
    this.dataService.getHosts(run, tag).subscribe(response => {
      let hosts = (response as string[]) || [''];
      if (hosts.length === 0) {
        hosts.push('');
      }
      hosts = hosts.map(host => {
        if (host === null) {
          return '';
        } else if (host === '') {
          return DEFAULT_HOST;
        }
        return host;
      });
      if (run === this.selectedRun && tag === this.selectedTag) {
        this.hosts = hosts;
        this.selectedHost = this.hosts[0];
        this.emitUpdateEvent();
      }
    });
  }

  emitUpdateEvent() {
    this.store.dispatch(setLoadingStateAction({
      loadingState: {
        loading: false,
        message: '',
      }
    }));

    this.update.emit({
      run: this.selectedRun,
      tag: this.selectedTag,
      host: this.selectedHost === DEFAULT_HOST ? '' : this.selectedHost
    });
  }
}
