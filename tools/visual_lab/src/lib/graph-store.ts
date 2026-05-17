import { writable } from 'svelte/store';
import type { GraphSnapshot } from './schema';
import { sampleGraphs } from './sample-bundle';

export const selectedGraph = writable<GraphSnapshot>(sampleGraphs[0]);
export const selectedNodeId = writable<string | null>(null);
export const selectedEdgeId = writable<string | null>(null);
export const selectedPocketId = writable<string | null>(null);
export const roleFilter = writable<string>('all');
