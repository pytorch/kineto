const path = require('path')
const HtmlWebpackPlugin = require('html-webpack-plugin')
const InlineChunkHtmlPlugin = require('inline-chunk-html-plugin')

const isDev = process.env.NODE_ENV !== 'production'

/**
 * @type {import('webpack').Configuration & import('webpack-dev-server').Configuration}
 */
module.exports = {
  mode: isDev ? 'development' : 'production',
  entry: './src/index.tsx',
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'index.js'
  },
  resolve: {
    // Add `.ts` and `.tsx` as a resolvable extension.
    extensions: ['.ts', '.tsx', '.js']
  },
  module: {
    rules: [
      { test: /\.tsx?$/i, use: 'ts-loader' },
      { test: /\.css$/i, use: ['style-loader', 'css-loader'] }
    ]
  },
  plugins: [
    new HtmlWebpackPlugin({
      inject: true,
      scriptLoading: 'blocking',
      template: 'index.html'
    }),
    !isDev ? new InlineChunkHtmlPlugin(HtmlWebpackPlugin, [/.*/]) : undefined
  ].filter(Boolean),
  devServer: {
    // proxy: {
    //     '/data/plugin/pytorch_profiler': ''
    // }
  }
}
