/**
 * 合并 part1.json（tiles 等）+ tileColors-import.json + furniture-import.json → 默认四份 + tiles-raw.txt
 */
import { readFileSync, writeFileSync } from 'fs'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const root = join(__dirname, '..')
const assets = join(root, 'assets')

const p1 = JSON.parse(readFileSync(join(__dirname, 'part1.json'), 'utf8'))
const tileColors = JSON.parse(readFileSync(join(__dirname, 'tileColors-import.json'), 'utf8'))
const furniture = JSON.parse(readFileSync(join(__dirname, 'furniture-import.json'), 'utf8'))

const layout = {
  ...p1,
  tileColors,
  furniture,
  layoutRevision: 4,
}

if (layout.tiles.length !== layout.cols * layout.rows) {
  console.error('tiles length', layout.tiles.length)
  process.exit(1)
}
if (tileColors.length !== layout.cols * layout.rows) {
  console.error('tileColors length', tileColors.length)
  process.exit(1)
}

const out = JSON.stringify(layout, null, 2)
writeFileSync(join(assets, 'default-layout-1.json'), out, 'utf8')
for (const name of ['boss-office-layout.json', 'office-scene-layout.json', 'library-scene-layout.json']) {
  writeFileSync(join(root, name), out, 'utf8')
}
writeFileSync(join(__dirname, 'tiles-raw.txt'), layout.tiles.join(','), 'utf8')
console.log('OK layoutRevision', layout.layoutRevision, 'tileColors', tileColors.length, 'furniture', furniture.length)
