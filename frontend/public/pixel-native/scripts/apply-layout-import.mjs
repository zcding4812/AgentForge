/**
 * 合并 tiles（tiles-raw.txt）+ tileColors-import.json + furniture-import.json → 默认布局四份。
 * 用法：node scripts/apply-layout-import.mjs
 */
import { readFileSync, writeFileSync } from 'fs'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const root = join(__dirname, '..')
const assets = join(root, 'assets')

const tiles = readFileSync(join(__dirname, 'tiles-raw.txt'), 'utf8')
  .trim()
  .split(',')
  .map((x) => parseInt(x.trim(), 10))
const tileColors = JSON.parse(readFileSync(join(__dirname, 'tileColors-import.json'), 'utf8'))
const furniture = JSON.parse(readFileSync(join(__dirname, 'furniture-import.json'), 'utf8'))

if (tiles.length !== 21 * 25) {
  console.error('tiles length', tiles.length, 'expected', 21 * 25)
  process.exit(1)
}
if (tileColors.length !== 21 * 25) {
  console.error('tileColors length', tileColors.length, 'expected', 21 * 25)
  process.exit(1)
}

const layout = {
  version: 1,
  cols: 21,
  rows: 25,
  layoutRevision: 3,
  tiles,
  tileColors,
  furniture,
}

const out = JSON.stringify(layout, null, 2)
writeFileSync(join(assets, 'default-layout-1.json'), out, 'utf8')
for (const name of ['boss-office-layout.json', 'office-scene-layout.json', 'library-scene-layout.json']) {
  writeFileSync(join(root, name), out, 'utf8')
}

console.log('OK layoutRevision', layout.layoutRevision, 'furniture', furniture.length)
