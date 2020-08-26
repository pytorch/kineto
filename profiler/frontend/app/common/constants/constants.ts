/** URL when running locally */
export const LOCAL_URL = '/';

/** Plugin name */
export const PLUGIN_NAME = 'profile';

/** Pefix of API */
export const API_PREFIX = '/data/plugin/';

/** Tools API */
export const TOOLS_API = API_PREFIX + PLUGIN_NAME + '/tools';

/** Hosts API */
export const HOSTS_API = API_PREFIX + PLUGIN_NAME + '/hosts';

/** Data API */
export const DATA_API = API_PREFIX + PLUGIN_NAME + '/data';

/** Capture Profile API */
export const CAPTURE_PROFILE_API =
    API_PREFIX + PLUGIN_NAME + '/capture_profile';

/** Default Host */
export const DEFAULT_HOST = 'default';

/** The map of podStatsRecord key and label */
export const POD_STATS_RECORD_PROPERTY_MAP:
    Array<{key: string, label: string}> = [
      {key: 'highFlopsComputeUs', label: 'High flops compute'},
      {key: 'lowFlopsComputeUs', label: 'Low flops compute'},
      {key: 'hostInfeedDurationUs', label: 'Host infeed'},
      {key: 'hostOutfeedDurationUs', label: 'Host outfeed'},
      {key: 'allReduceComputeDurationUs', label: 'AllReduce compute'},
      {key: 'allReduceSyncDurationUs', label: 'AllReduce sync'},
      {key: 'sendDurationUs', label: 'Send'},
      {key: 'recvDurationUs', label: 'Recv'},
    ];
