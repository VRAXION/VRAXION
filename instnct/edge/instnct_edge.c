/*
 * INSTNCT Edge — Forward pass implementation
 * Pure integer arithmetic. No float. No sin().
 */

#include "instnct_edge.h"
#include <string.h>

void instnct_state_reset(InstnctState *state) {
    memset(state->charge, 0, sizeof(state->charge));
    memset(state->refr_counter, 0, sizeof(state->refr_counter));
}

void instnct_rollout_token(
    const InstnctNeuron neurons[],
    const InstnctEdge edges[],
    int n_edges,
    const uint16_t sdr_active[],
    int n_active,
    InstnctState *state,
    int8_t spike_out[]
) {
    int8_t act[INSTNCT_H];
    int16_t raw[INSTNCT_H];
    int i, e, k, tick;
    uint8_t decay_counter = 0;

    memset(act, 0, sizeof(act));

    for (tick = 0; tick < INSTNCT_TICKS; tick++) {

        /* 1. DECAY: subtract 1 every DECAY_PERIOD ticks */
        decay_counter++;
        if (decay_counter >= INSTNCT_DECAY_PERIOD) {
            for (i = 0; i < INSTNCT_H; i++) {
                if (state->charge[i] > 0)
                    state->charge[i]--;
            }
            decay_counter = 0;
        }

        /* 2. INPUT: inject SDR for first INPUT_DURATION ticks */
        if (tick < INSTNCT_INPUT_DURATION) {
            for (k = 0; k < n_active; k++) {
                if (sdr_active[k] < INSTNCT_H)
                    act[sdr_active[k]] = 1;
            }
        }

        /* 3. PROPAGATE: only through existing connections, only from active neurons */
        memset(raw, 0, sizeof(raw));
        for (e = 0; e < n_edges; e++) {
            if (act[edges[e].source] != 0) {
                int8_t val = neurons[edges[e].source].polarity ?
                             act[edges[e].source] : -act[edges[e].source];
                raw[edges[e].target] += val;
            }
        }

        /* Add propagated charge and clamp */
        for (i = 0; i < INSTNCT_H; i++) {
            int16_t new_charge = (int16_t)state->charge[i] + raw[i];
            if (new_charge < 0) new_charge = 0;
            if (new_charge > INSTNCT_MAX_CHARGE) new_charge = INSTNCT_MAX_CHARGE;
            state->charge[i] = (int8_t)new_charge;
        }

        /* 4. SPIKE DECISION with per-neuron refractory + channel LUT */
        for (i = 0; i < INSTNCT_H; i++) {
            if (state->refr_counter[i] > 0) {
                /* Still in refractory — can't fire */
                state->refr_counter[i]--;
                spike_out[i] = 0;
            } else {
                /* Check threshold from precomputed LUT */
                uint8_t eff_th = INSTNCT_EFF_THETA[neurons[i].theta]
                                                   [neurons[i].channel]
                                                   [tick];
                if (state->charge[i] >= (int8_t)eff_th) {
                    /* FIRE! */
                    spike_out[i] = 1;
                    state->charge[i] = 0;
                    state->refr_counter[i] = neurons[i].refr_period;
                } else {
                    spike_out[i] = 0;
                }
            }
        }

        /* 5. UPDATE ACT: spike × polarity -> {-1, 0, +1} */
        for (i = 0; i < INSTNCT_H; i++) {
            if (spike_out[i]) {
                act[i] = neurons[i].polarity ? 1 : -1;
            } else {
                act[i] = 0;
            }
        }
    }
}

uint8_t instnct_readout_simple(const InstnctState *state) {
    /* Simple readout: find the output neuron with highest charge.
     * Output zone: last OUT_DIM neurons.
     * Map neuron index to byte value (0-255). */
    int out_start = INSTNCT_H - (int)(INSTNCT_H / 1.618);  /* H - H/phi */
    int out_dim = INSTNCT_H - out_start;
    int best_idx = 0;
    int8_t best_val = -1;
    int i;

    for (i = 0; i < out_dim; i++) {
        if (state->charge[out_start + i] > best_val) {
            best_val = state->charge[out_start + i];
            best_idx = i;
        }
    }

    /* Map output neuron index to byte: proportional */
    return (uint8_t)((best_idx * 256) / out_dim);
}
