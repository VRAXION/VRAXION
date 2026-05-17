import { writable } from 'svelte/store';

export const checkpoint = writable(0);
export const tick = writable(0);
export const playing = writable(false);
