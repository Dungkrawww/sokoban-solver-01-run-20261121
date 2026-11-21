// Sokoban solver: push-optimal per room.
//
// The generator builds each level as one hallway (the player's row) plus 1-4
// independent rooms, each opening onto the hallway through a single door, and
// a box pushed out of a room can never come back - so every box stays in its
// room and a level's push-optimal count is the sum of its rooms' optima.
// This solver decomposes the level into rooms, solves each room push-optimally
// with A* over pushes (admissible min-assignment heuristic + deadlock pruning),
// and stitches the rooms together with hallway walks. Deterministic: no clock,
// no randomness, expansion caps only - two runs write identical bytes.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
// [dx, dy, letter] - fixed order everywhere, so every search is deterministic.
const DIRS: [number, number, string][] = [[0, -1, 'u'], [0, 1, 'd'], [-1, 0, 'l'], [1, 0, 'r']];

interface Level {
  w: number; h: number; wall: boolean[]; goals: boolean[]; boxes: number[]; player: number;
}

function parse(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Array<boolean>(w * h).fill(true);
  const goals = new Array<boolean>(w * h).fill(false);
  const boxes: number[] = [];
  let player = 0;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      wall[i] = ch === '#';
      if ('.*+'.includes(ch)) goals[i] = true;
      if ('$*'.includes(ch)) boxes.push(i);
      if ('@+'.includes(ch)) player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

// Breadth-first distances from `start` over cells where pass(i) holds.
function bfs(start: number, pass: (i: number) => boolean, w: number, h: number): Int32Array {
  const d = new Int32Array(w * h).fill(-1);
  d[start] = 0;
  const q = [start];
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    const cx = c % w, cy = Math.floor(c / w);
    for (const [dx, dy] of DIRS) {
      const nx = cx + dx, ny = cy + dy;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const n = ny * w + nx;
      if (d[n] >= 0 || !pass(n)) continue;
      d[n] = d[c]! + 1;
      q.push(n);
    }
  }
  return d;
}

// Shortest walk (as a move string) from `from` to `to` over passable cells, or '' if unreachable.
function walkPath(from: number, to: number, pass: (i: number) => boolean, w: number, h: number): string {
  if (from === to) return '';
  const prev = new Int32Array(w * h).fill(-1);
  const step = new Array<string>(w * h).fill('');
  prev[from] = from;
  const q = [from];
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    const cx = c % w, cy = Math.floor(c / w);
    for (const [dx, dy, letter] of DIRS) {
      const nx = cx + dx, ny = cy + dy;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const n = ny * w + nx;
      if (prev[n] >= 0 || !pass(n)) continue;
      prev[n] = c; step[n] = letter;
      if (n === to) {
        let s = '', t = to;
        while (t !== from) { s = step[t] + s; t = prev[t]!; }
        return s;
      }
      q.push(n);
    }
  }
  return '';
}

// Cells dead for a box: a box here can never reach a goal.
//  - corner: wall on two perpendicular sides, not a goal.
//  - wall run: a maximal run of floor cells sliding along one wall, with no
//    goal in the run and both ends blocked by walls (a box in it can only
//    slide along the run and can never leave it).
function deadCells(cells: number[], wall: boolean[], goals: boolean[], w: number, h: number): boolean[] {
  const inSet = new Set(cells);
  const isWall = (x: number, y: number): boolean =>
    x < 0 || y < 0 || x >= w || y >= h || wall[y * w + x] || !inSet.has(y * w + x);
  const dead = new Map<number, boolean>();
  for (const c of cells) {
    if (goals[c]) { dead.set(c, false); continue; }
    const x = c % w, y = Math.floor(c / w);
    const up = isWall(x, y - 1), dn = isWall(x, y + 1), lf = isWall(x - 1, y), rt = isWall(x + 1, y);
    dead.set(c, (up || dn) && (lf || rt));
  }
  // Horizontal runs: consecutive cells with a wall directly above or below.
  for (let y = 0; y < h; y++) {
    let run: number[] = [];
    let startBlocked = true;
    const flush = (endBlocked: boolean): void => {
      if (run.length > 0 && startBlocked && endBlocked && !run.some((r) => goals[r])) {
        for (const r of run) dead.set(r, true);
      }
      run = [];
    };
    for (let x = 0; x <= w; x++) {
      const c = y * w + x;
      const inRun = x < w && inSet.has(c) && (isWall(x, y - 1) || isWall(x, y + 1));
      if (inRun) {
        if (run.length === 0) startBlocked = x === 0 || !inSet.has(c - 1) || wall[c - 1];
        run.push(c);
        continue;
      }
      // The run ends here: blocked when the next cell is not passable floor.
      flush(x >= w || !inSet.has(c) || wall[c]);
    }
  }
  // Vertical runs: consecutive cells with a wall directly left or right.
  for (let x = 0; x < w; x++) {
    let run: number[] = [];
    let startBlocked = true;
    const flush = (endBlocked: boolean): void => {
      if (run.length > 0 && startBlocked && endBlocked && !run.some((r) => goals[r])) {
        for (const r of run) dead.set(r, true);
      }
      run = [];
    };
    for (let y = 0; y <= h; y++) {
      const c = y * w + x;
      const inRun = y < h && inSet.has(c) && (isWall(x - 1, y) || isWall(x + 1, y));
      if (inRun) {
        if (run.length === 0) startBlocked = y === 0 || !inSet.has(c - w) || wall[c - w];
        run.push(c);
        continue;
      }
      flush(y >= h || !inSet.has(c) || wall[c]);
    }
  }
  return cells.map((c) => dead.get(c) ?? false);
}

interface PushProblem {
  cells: number[]; // local floor, incl. entry cell
  startPlayer: number; // global cell index
  boxes: number[]; // global cell indices, sorted
  goals: boolean[]; // global
  exitCell: number | null; // global cell the player must be able to reach at the end (door), or null
  wall: boolean[]; w: number; h: number;
}

// A* over pushes. Returns the move string (lowercase walks, uppercase pushes) or null.
// weight 1 = optimal; higher weights trade optimality for speed. Deterministic cap.
function solvePushes(prob: PushProblem, cap: number, weight: number): string | null {
  const { cells, wall, goals, w, h } = prob;
  const n = cells.length;
  const local = new Map<number, number>();
  cells.forEach((c, i) => local.set(c, i));
  const isLocal = (c: number): boolean => local.has(c);
  const dead = deadCells(cells, wall, goals, w, h);
  const goalLocal = new Set<number>();
  cells.forEach((c, i) => { if (goals[c]) goalLocal.add(i); });
  // All-pairs distances over local floor (ignoring boxes).
  const dist: Int32Array[] = [];
  for (let i = 0; i < n; i++) {
    const d = bfs(cells[i]!, (c) => isLocal(c), w, h);
    const row = new Int32Array(n);
    for (let j = 0; j < n; j++) row[j] = d[cells[j]!]!;
    dist.push(row);
  }
  const startBoxes = prob.boxes.map((c) => local.get(c)!).sort((a, b) => a - b);
  const startPlayer = local.get(prob.startPlayer)!;
  const exitLocal = prob.exitCell === null ? -1 : local.get(prob.exitCell);

  // Reachable region of the player given boxes; returns a bitset as sorted array + canonical min cell.
  const region = (player: number, boxSet: Set<number>): { cells: number[]; canon: number } => {
    const seen = new Set<number>([player]);
    const q = [player];
    for (let qi = 0; qi < q.length; qi++) {
      const c = q[qi]!;
      const cx = cells[c]! % w, cy = Math.floor(cells[c]! / w);
      for (const [dx, dy] of DIRS) {
        const nx = cx + dx, ny = cy + dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const g = ny * w + nx;
        const li = local.get(g);
        if (li === undefined || seen.has(li) || boxSet.has(li)) continue;
        seen.add(li); q.push(li);
      }
    }
    const arr = [...seen].sort((a, b) => a - b);
    return { cells: arr, canon: arr[0]! };
  };

  // Min-assignment heuristic: min over permutations of sum of distances box->goal.
  const heuristic = (bs: number[]): number => {
    const gs = [...goalLocal];
    if (bs.length !== gs.length) return 1e9;
    let best = Infinity;
    const perm = gs.slice();
    const rec = (k: number, acc: number): void => {
      if (acc >= best) return;
      if (k === bs.length) { best = acc; return; }
      for (let i = k; i < perm.length; i++) {
        [perm[k], perm[i]] = [perm[i]!, perm[k]!];
        const d = dist[bs[k]!]![perm[k]!]!;
        if (d >= 0) rec(k + 1, acc + d);
        [perm[k], perm[i]] = [perm[i]!, perm[k]!];
      }
    };
    rec(0, 0);
    return best === Infinity ? 1e9 : best;
  };

  const keyOf = (bs: number[], canon: number): string => bs.join(',') + '|' + canon;
  const startRegion = region(startPlayer, new Set(startBoxes));
  const startKey = keyOf(startBoxes, startRegion.canon);
  // Node storage.
  const nodeBoxes: number[][] = [startBoxes];
  const nodePlayer: number[] = [startPlayer];
  const nodeG: number[] = [0];
  const nodeParent: number[] = [-1];
  const nodePush: [number, number][] = [[-1, -1]]; // [box local index before push, dir index]
  const best = new Map<string, number>([[startKey, 0]]);
  // Binary heap of [f, -g, nodeId]; ties prefer deeper g (find a goal fast).
  const heap: [number, number, number][] = [[heuristic(startBoxes) * weight, 0, 0]];
  const less = (a: [number, number, number], b: [number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const hpush = (e: [number, number, number]): void => {
    let i = heap.length; heap.push(e);
    while (i > 0) { const p = (i - 1) >> 1; if (!less(heap[i]!, heap[p]!)) break; [heap[i], heap[p]] = [heap[p]!, heap[i]!]; i = p; }
  };
  const hpop = (): [number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!]; i = c;
      }
    }
    return top;
  };

  let expanded = 0;
  let goalNode = -1;
  while (heap.length > 0 && expanded < cap) {
    const [, , s] = hpop();
    const bs = nodeBoxes[s]!;
    const reg = region(nodePlayer[s]!, new Set(bs));
    if (best.get(keyOf(bs, reg.canon)) !== s) continue;
    // Goal: every box on a goal, and the exit reachable when required.
    if (bs.every((b) => goalLocal.has(b)) && (exitLocal < 0 || reg.cells.includes(exitLocal))) { goalNode = s; break; }
    expanded++;
    const boxSet = new Set(bs);
    for (let bi = 0; bi < bs.length; bi++) {
      const b = bs[bi]!;
      const bx = cells[b]! % w, by = Math.floor(cells[b]! / w);
      for (let di = 0; di < 4; di++) {
        const [dx, dy] = DIRS[di]!;
        const tx = bx + dx, ty = by + dy;
        if (tx < 0 || ty < 0 || tx >= w || ty >= h) continue;
        const tgt = local.get(ty * w + tx);
        if (tgt === undefined || boxSet.has(tgt)) continue;
        if (dead[tgt] && !goalLocal.has(tgt)) continue;
        const behind = local.get((by - dy) * w + (bx - dx));
        if (behind === undefined || boxSet.has(behind)) continue;
        if (!reg.cells.includes(behind)) continue;
        // 2x2 freeze: a 2x2 block of walls/boxes with a non-goal box is dead.
        const nb = bs.slice(); nb[bi] = tgt; nb.sort((a, b) => a - b);
        const nbSet = new Set(nb);
        let frozen = false;
        for (const q of nb) {
          if (goalLocal.has(q)) continue;
          const qx = cells[q]! % w, qy = Math.floor(cells[q]! / w);
          for (const [ox, oy] of [[0, 0], [-1, 0], [0, -1], [-1, -1]] as [number, number][]) {
            const cx0 = qx + ox, cy0 = qy + oy;
            let block = true, hasBox = false;
            for (const [ax, ay] of [[0, 0], [1, 0], [0, 1], [1, 1]] as [number, number][]) {
              const gx = cx0 + ax, gy = cy0 + ay;
              const gi = gy * w + gx;
              const li = local.get(gi);
              if (li !== undefined && nbSet.has(li)) { hasBox = true; continue; }
              if (gx < 0 || gy < 0 || gx >= w || gy >= h || wall[gi] || !isLocal(gi)) continue;
              block = false; break;
            }
            if (block && hasBox) { frozen = true; break; }
          }
          if (frozen) break;
        }
        if (frozen) continue;
        const nReg = region(b, nbSet);
        const k = keyOf(nb, nReg.canon);
        const seen = best.get(k);
        if (seen !== undefined && nodeG[seen]! <= nodeG[s]! + 1) continue;
        const t = nodeBoxes.length;
        nodeBoxes.push(nb); nodePlayer.push(b); nodeG.push(nodeG[s]! + 1);
        nodeParent.push(s); nodePush.push([b, di]);
        best.set(k, t);
        const hh = heuristic(nb);
        if (hh >= 1e9) continue;
        hpush([nodeG[t]! + weight * hh, -nodeG[t]!, t]);
      }
    }
  }
  if (goalNode < 0) return null;
  // Reconstruct the push list, then expand each push into walk + push moves.
  const pushes: [number, number][] = [];
  for (let t = goalNode; t > 0; t = nodeParent[t]!) pushes.push(nodePush[t]!);
  pushes.reverse();
  let moves = '';
  let curPlayer = startPlayer;
  let curBoxes = startBoxes.slice();
  for (const [boxLocal, di] of pushes) {
    const [dx, dy] = DIRS[di]!;
    const bGlobal = cells[boxLocal]!;
    const bx = bGlobal % w, by = Math.floor(bGlobal / w);
    const behindGlobal = (by - dy) * w + (bx - dx);
    const behindLocal = local.get(behindGlobal)!;
    const boxSet = new Set(curBoxes);
    // Walk on local indices: BFS from curPlayer to behindLocal avoiding boxes.
    const prev = new Int32Array(n).fill(-1);
    const stp = new Array<string>(n).fill('');
    prev[curPlayer] = curPlayer;
    const q = [curPlayer];
    let found = false;
    for (let qi = 0; qi < q.length && !found; qi++) {
      const c = q[qi]!;
      const cx = cells[c]! % w, cy = Math.floor(cells[c]! / w);
      for (let dj = 0; dj < 4; dj++) {
        const [ddx, ddy, letter] = DIRS[dj]!;
        const nx = cx + ddx, ny = cy + ddy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const li = local.get(ny * w + nx);
        if (li === undefined || prev[li] >= 0 || boxSet.has(li)) continue;
        prev[li] = c; stp[li] = letter;
        if (li === behindLocal) { found = true; break; }
        q.push(li);
      }
    }
    if (!found && curPlayer !== behindLocal) return null;
    let seg = '';
    let t = behindLocal;
    while (t !== curPlayer) { seg = stp[t] + seg; t = prev[t]!; }
    moves += seg + DIRS[di]![2].toUpperCase();
    curPlayer = boxLocal;
    const idx = curBoxes.indexOf(boxLocal);
    const tgtLocal = local.get((by + dy) * w + (bx + dx))!;
    curBoxes[idx] = tgtLocal;
    curBoxes.sort((a, b) => a - b);
  }
  return moves;
}

// Move-based A* fallback (like the house baseline): minimises moves, not pushes.
// Only used when push search fails; any valid solution beats an empty one.
function solveMoves(prob: PushProblem, cap: number): string | null {
  const { cells, wall, goals, w, h } = prob;
  const local = new Map<number, number>();
  cells.forEach((c, i) => local.set(c, i));
  const goalLocal = new Set<number>();
  cells.forEach((c, i) => { if (goals[c]) goalLocal.add(i); });
  const startBoxes = prob.boxes.map((c) => local.get(c)!).sort((a, b) => a - b);
  const startPlayer = local.get(prob.startPlayer)!;
  const distToGoal = (b: number): number => {
    let m = Infinity;
    for (const g of goalLocal) {
      const d = Math.abs((cells[g]! % w) - (cells[b]! % w)) + Math.abs(Math.floor(cells[g]! / w) - Math.floor(cells[b]! / w));
      if (d < m) m = d;
    }
    return m;
  };
  const est = (bs: number[]): number => bs.reduce((a, b) => a + distToGoal(b), 0);
  const keyOf = (p: number, bs: number[]): string => p + '|' + bs.join(',');
  const sp: number[] = [startPlayer];
  const sb: number[][] = [startBoxes];
  const sg: number[] = [0];
  const parent: number[] = [-1];
  const step: string[] = [''];
  const best = new Map<string, number>([[keyOf(startPlayer, startBoxes), 0]]);
  const heap: [number, number, number][] = [[est(startBoxes), 0, 0]];
  const less = (a: [number, number, number], b: [number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const hpush = (e: [number, number, number]): void => {
    let i = heap.length; heap.push(e);
    while (i > 0) { const p = (i - 1) >> 1; if (!less(heap[i]!, heap[p]!)) break; [heap[i], heap[p]] = [heap[p]!, heap[i]!]; i = p; }
  };
  const hpop = (): [number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!]; i = c;
      }
    }
    return top;
  };
  let expanded = 0;
  while (heap.length > 0 && expanded < cap) {
    const [, g, s] = hpop();
    if (best.get(keyOf(sp[s]!, sb[s]!)) !== s) continue;
    const bs = sb[s]!;
    if (bs.every((b) => goalLocal.has(b))) {
      let moves = '';
      for (let t = s; t > 0; t = parent[t]!) moves = step[t] + moves;
      return moves;
    }
    expanded++;
    const p = sp[s]!;
    const px = cells[p]! % w, py = Math.floor(cells[p]! / w);
    for (const [dx, dy, letter] of DIRS) {
      const nx = px + dx, ny = py + dy;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const q = local.get(ny * w + nx);
      if (q === undefined) continue;
      let nb = bs;
      let mv = letter;
      const at = bs.indexOf(q);
      if (at >= 0) {
        const tx = nx + dx, ty = ny + dy;
        if (tx < 0 || ty < 0 || tx >= w || ty >= h) continue;
        const tgt = local.get(ty * w + tx);
        if (tgt === undefined || bs.includes(tgt)) continue;
        nb = bs.slice(); nb[at] = tgt; nb.sort((a, b) => a - b);
        mv = letter.toUpperCase();
      }
      const k = keyOf(q, nb);
      const seen = best.get(k);
      if (seen !== undefined && sg[seen]! <= g + 1) continue;
      const t = sp.length;
      sp.push(q); sb.push(nb); sg.push(g + 1); parent.push(s); step.push(mv);
      best.set(k, t);
      hpush([g + 1 + est(nb), g + 1, t]);
    }
  }
  return null;
}

interface Room { cells: number[]; door: number; entry: number; boxes: number[]; }

function findRooms(lv: Level): Room[] | null {
  const { w, h, wall, boxes } = lv;
  const hallRow = Math.floor(lv.player / w);
  const seen = new Set<number>();
  const rooms: Room[] = [];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (y === hallRow || wall[i] || seen.has(i)) continue;
      const q = [i]; seen.add(i);
      const comp: number[] = [];
      while (q.length) {
        const c = q.pop()!;
        comp.push(c);
        const cx = c % w, cy = Math.floor(c / w);
        for (const [dx, dy] of DIRS) {
          const nx = cx + dx, ny = cy + dy;
          if (ny === hallRow || nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          const n = ny * w + nx;
          if (wall[n] || seen.has(n)) continue;
          seen.add(n); q.push(n);
        }
      }
      // The door: the component cell adjacent to the hallway row.
      const doors = comp.filter((c) => Math.abs(Math.floor(c / w) - hallRow) === 1 && !wall[(hallRow * w) + (c % w)]);
      if (doors.length !== 1) return null;
      const door = doors[0]!;
      const entry = hallRow * w + (door % w);
      const rboxes = boxes.filter((b) => comp.includes(b));
      const rgoals = comp.filter((c) => lv.goals[c]);
      if (rboxes.length === 0 || rboxes.length !== rgoals.length) return null;
      rooms.push({ cells: comp, door, entry, boxes: rboxes });
    }
  }
  if (rooms.length === 0) return null;
  // Every box must belong to exactly one room.
  const inRoom = new Set(rooms.flatMap((r) => r.boxes));
  if (inRoom.size !== boxes.length) return null;
  rooms.sort((a, b) => a.door - b.door);
  return rooms;
}

// Replay a move string on the level; returns the final boxes/player, or null on an illegal step.
function replay(lv: Level, moves: string): { boxes: number[]; player: number } | null {
  const { w, h, wall } = lv;
  let boxes = lv.boxes.slice();
  let player = lv.player;
  for (const ch of moves) {
    const l = ch.toLowerCase();
    const di = DIRS.findIndex((d) => d[2] === l);
    if (di < 0) return null;
    const [dx, dy] = DIRS[di]!;
    const px = player % w, py = Math.floor(player / w);
    const nx = px + dx, ny = py + dy;
    if (nx < 0 || ny < 0 || nx >= w || ny >= h) return null;
    const q = ny * w + nx;
    if (wall[q]) return null;
    const at = boxes.indexOf(q);
    if (at >= 0) {
      const tx = nx + dx, ty = ny + dy;
      if (tx < 0 || ty < 0 || tx >= w || ty >= h) return null;
      const tgt = ty * w + tx;
      if (wall[tgt] || boxes.includes(tgt)) return null;
      boxes[at] = tgt;
      boxes.sort((a, b) => a - b);
    }
    player = q;
  }
  return { boxes, player };
}

function countPushes(moves: string): number {
  let n = 0;
  for (const c of moves) if (c >= 'A' && c <= 'Z') n++;
  return n;
}

// Solve one room (floor = room cells + entry cell). Tries exact A*, then
// weighted A*, then a move-based search. Returns the move string or null.
function solveRoom(cur: Level, room: Room, needExit: boolean): string | null {
  const prob: PushProblem = {
    cells: [...room.cells, room.entry],
    startPlayer: room.entry,
    boxes: room.boxes,
    goals: cur.goals,
    exitCell: needExit ? room.door : null,
    wall: cur.wall, w: cur.w, h: cur.h,
  };
  let sol: string | null = null;
  for (const [cap, weight] of [[600_000, 1], [600_000, 2], [600_000, 4]] as [number, number][]) {
    sol = solvePushes(prob, cap, weight);
    if (sol !== null) break;
  }
  if (sol === null) sol = solveMoves(prob, 500_000);
  return sol;
}

function solveLevel(text: string): string {
  const lv = parse(text);
  const rooms = findRooms(lv);
  if (rooms) {
    // A room's push count does not depend on visit order (boxes never leave
    // their room), but only the last room visited may end with the player
    // unable to leave - so solve every room both ways and pick the last room
    // that minimises the total pushes.
    const exitSols = rooms.map((room) => solveRoom(lv, room, true));
    const freeSols = rooms.map((room) => solveRoom(lv, room, false));
    let last = -1;
    let bestTotal = Infinity;
    for (let l = 0; l < rooms.length; l++) {
      if (freeSols[l] === null) continue;
      let total = countPushes(freeSols[l]!);
      let ok = true;
      for (let i = 0; i < rooms.length; i++) {
        if (i === l) continue;
        if (exitSols[i] === null) { ok = false; break; }
        total += countPushes(exitSols[i]!);
      }
      if (!ok) continue;
      // Tie-break: prefer the last room in door order, keeping the visit order natural.
      if (total < bestTotal || (total === bestTotal && l > last)) { bestTotal = total; last = l; }
    }
    const order = rooms.map((_, i) => i).filter((i) => i !== last);
    if (last >= 0) order.push(last);
    let moves = '';
    let cur: Level = lv;
    for (let oi = 0; oi < order.length; oi++) {
      const ri = order[oi]!;
      const room = rooms[ri]!;
      const isLast = oi === order.length - 1;
      const sol = isLast ? (freeSols[ri] ?? exitSols[ri]) : exitSols[ri];
      if (sol === null || sol === undefined) return moves;
      // Walk the player to the room's entry cell (hallway is box-free).
      const boxSet = new Set(cur.boxes);
      const walk = walkPath(cur.player, room.entry, (c) => !cur.wall[c] && !boxSet.has(c), cur.w, cur.h);
      const afterWalk = replay(cur, walk);
      if (!afterWalk) return moves;
      cur = { ...cur, boxes: afterWalk.boxes, player: afterWalk.player };
      moves += walk;
      const afterRoom = replay(cur, sol);
      if (!afterRoom) return moves;
      cur = { ...cur, boxes: afterRoom.boxes, player: afterRoom.player };
      moves += sol;
    }
    return moves;
  }
  // Fallback: whole level as one push problem (no decomposition).
  const allFloor: number[] = [];
  for (let i = 0; i < lv.w * lv.h; i++) if (!lv.wall[i]) allFloor.push(i);
  const prob: PushProblem = {
    cells: allFloor, startPlayer: lv.player, boxes: lv.boxes, goals: lv.goals,
    exitCell: null, wall: lv.wall, w: lv.w, h: lv.h,
  };
  return solvePushes(prob, 1_000_000, 1) ?? solveMoves(prob, 500_000) ?? '';
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
