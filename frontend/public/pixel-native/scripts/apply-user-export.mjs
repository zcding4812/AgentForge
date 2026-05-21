/**
 * 将本目录下 user-layout-full.json（浏览器导出的完整布局）写入默认四份，并同步 tiles-raw.txt。
 * 用法：node scripts/apply-user-export.mjs
 */
import { readFileSync, writeFileSync } from 'fs'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const root = join(__dirname, '..')
const assets = join(root, 'assets')
const src = join(__dirname, 'user-layout-full.json')

const u = JSON.parse(readFileSync(src, 'utf8'))
const layout = {
  version: 1,
  cols: u.cols,
  rows: u.rows,
  layoutRevision: 4,
  tiles: u.tiles,
  tileColors: u.tileColors,
  furniture: u.furniture,
}

if (layout.tiles.length !== layout.cols * layout.rows) {
  console.error('tiles length mismatch', layout.tiles.length, layout.cols * layout.rows)
  process.exit(1)
}
if (layout.tileColors.length !== layout.cols * layout.rows) {
  console.error('tileColors length mismatch', layout.tileColors.length, layout.cols * layout.rows)
  process.exit(1)
}

const out = JSON.stringify(layout, null, 2)
writeFileSync(join(assets, 'default-layout-1.json'), out, 'utf8')
for (const name of ['boss-office-layout.json', 'office-scene-layout.json', 'library-scene-layout.json']) {
  writeFileSync(join(root, name), out, 'utf8')
}

writeFileSync(join(__dirname, 'tiles-raw.txt'), layout.tiles.join(','), 'utf8')

console.log('OK layoutRevision', layout.layoutRevision, 'tiles', layout.tiles.length, 'furniture', layout.furniture.length)
