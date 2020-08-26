import {Component, OnInit} from '@angular/core';
import {compareTagNames} from 'org_xprof/frontend/app/common/classes/sorting';
import {Tool} from 'org_xprof/frontend/app/common/interfaces/tool';
import {DataService} from 'org_xprof/frontend/app/services/data_service/data_service';

const RELOAD_INTERVAL_SECONDS = 30;
const THRESHOLD = 2;

/** The root component. */
@Component({
  selector: 'app',
  templateUrl: './app.ng.html',
  styleUrls: ['./app.css'],
})
export class App implements OnInit {
  loading = true;
  comparing = false;
  dataFound = false;
  datasets: Tool[] = [];
  prviousReloadTime = 0;

  constructor(private readonly dataService: DataService) {
    document.addEventListener('tensorboard-reload', () => {
      const currentReloadTime = new Date().getTime();
      const diff = (currentReloadTime - this.prviousReloadTime) / 1000;
      if (diff >= RELOAD_INTERVAL_SECONDS - THRESHOLD) {
        this.prviousReloadTime = currentReloadTime;
      }
      if (!this.loading && !this.comparing &&
          (diff < RELOAD_INTERVAL_SECONDS - THRESHOLD ||
           diff > RELOAD_INTERVAL_SECONDS + THRESHOLD)) {
        this.compareDatasets();
      }
    });
  }

  private processTools(tools: {[key: string]: string[]}): Tool[] {
    const datasets: Tool[] = [];
    const keys = Object.keys(tools);
    const values = Object.values(tools);
    for (let i = 0; i < keys.length; i++) {
      datasets.push({name: keys[i], activeTools: values[i] || []});
    }
    datasets.sort((a, b) => -compareTagNames(a.name, b.name));
    return datasets;
  }

  ngOnInit() {
    this.dataService.getTools().subscribe(tools => {
      this.datasets =
          this.processTools((tools || {}) as {[key: string]: string[]});
      this.dataFound = this.datasets.length !== 0;
      this.loading = false;
    });
  }

  compareDatasets() {
    this.comparing = true;
    this.dataService.getTools().subscribe(tools => {
      const datasets: Tool[] =
          this.processTools((tools || {}) as {[key: string]: string[]});
      if (JSON.stringify(datasets) !== JSON.stringify(this.datasets)) {
        document.dispatchEvent(new Event('plugin-reload'));
      }
      this.comparing = false;
    });
  }
}
