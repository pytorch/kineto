const fs = require('fs')
const path = require('path')

fs.copyFileSync(
  path.resolve(__dirname, 'dist/index.html'),
  path.resolve(
    __dirname,
    '../tensorboard_plugin_torch_profiler/static/index.html'
  )
)

console.log('Copy done.')
