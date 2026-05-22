import { writable } from 'svelte/store';
import type { MetricRow } from './schema';
import { sampleMetrics } from './sample-bundle';

export const metrics = writable<MetricRow[]>(sampleMetrics);
