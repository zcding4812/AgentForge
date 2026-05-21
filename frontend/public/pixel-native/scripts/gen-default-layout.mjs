/**
 * 从 scripts/tiles-raw.txt（逗号分隔）生成默认布局 JSON。
 * 运行：node scripts/gen-default-layout.mjs
 */
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const rootDir = path.join(__dirname, '..')
const assetsDir = path.join(rootDir, 'assets')

const raw = fs.readFileSync(path.join(__dirname, 'tiles-raw.txt'), 'utf8').trim()
const tiles = raw.split(',').map((x) => parseInt(x.trim(), 10))

const cols = 21
const rows = 25
if (tiles.length !== cols * rows) {
  console.error('tiles length', tiles.length, 'expected', cols * rows)
  process.exit(1)
}

const tileColors = Array(cols * rows).fill(null)

const prevPath = path.join(assetsDir, 'default-layout-1.json')
const prev = JSON.parse(fs.readFileSync(prevPath, 'utf8'))

const layout = {
  version: 1,
  cols,
  rows,
  layoutRevision: (prev.layoutRevision ?? 1) + 1,
  tiles,
  tileColors,
  furniture: prev.furniture,
}

const out = JSON.stringify(layout, null, 2)
const targets = [
  path.join(assetsDir, 'default-layout-1.json'),
  path.join(rootDir, 'boss-office-layout.json'),
  path.join(rootDir, 'office-scene-layout.json'),
  path.join(rootDir, 'library-scene-layout.json'),
]
for (const p of targets) {
  fs.writeFileSync(p, out, 'utf8')
}

console.log('OK layoutRevision', layout.layoutRevision, 'tiles', tiles.length, 'furniture', layout.furniture.length)
