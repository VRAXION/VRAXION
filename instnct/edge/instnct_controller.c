/*
 * INSTNCT Edge — ReLU controller implementation
 * All int8/int16 arithmetic. Zero float.
 *
 * Weight format: int8 scaled by 64 (so 1.0 = 64, -0.5 = -32).
 * Activations: int16 to avoid overflow during matmul.
 * ReLU: max(0, x) — single compare, no multiply.
 */

#include "instnct_controller.h"

void instnct_ctrl_scores(
    const InstnctController *ctrl,
    const InstnctCtrlInput *input,
    int16_t scores[CTRL_OUTPUT]
) {
    int16_t x[CTRL_INPUT];
    int16_t h1[CTRL_HIDDEN];
    int16_t h2[CTRL_HIDDEN];
    int16_t h3[CTRL_HIDDEN];
    int i, j;
    int32_t acc;

    /* Pack input: uint8 [0-100] -> int16 */
    x[0] = (int16_t)input->accuracy;
    x[1] = (int16_t)input->pain;
    x[2] = (int16_t)input->reward;
    x[3] = (int16_t)input->rate_add;
    x[4] = (int16_t)input->rate_flip;
    x[5] = (int16_t)input->rate_theta;
    x[6] = (int16_t)input->rate_channel;
    x[7] = (int16_t)input->rate_reverse;
    x[8] = (int16_t)input->rate_remove;
    x[9] = (int16_t)input->progress;

    /* Layer 1: x[10] @ W1[10×64] + b1[64] -> ReLU -> h1[64] */
    for (j = 0; j < CTRL_HIDDEN; j++) {
        acc = (int32_t)ctrl->b1[j] << 6;  /* bias scaled up */
        for (i = 0; i < CTRL_INPUT; i++) {
            acc += (int32_t)x[i] * (int32_t)ctrl->W1[i][j];
        }
        acc >>= 6;  /* scale back down */
        h1[j] = (acc > 0) ? (int16_t)acc : 0;  /* ReLU */
    }

    /* Layer 2: h1[64] @ W2[64×64] + b2[64] -> ReLU -> h2[64] */
    for (j = 0; j < CTRL_HIDDEN; j++) {
        acc = (int32_t)ctrl->b2[j] << 6;
        for (i = 0; i < CTRL_HIDDEN; i++) {
            acc += (int32_t)h1[i] * (int32_t)ctrl->W2[i][j];
        }
        acc >>= 6;
        h2[j] = (acc > 0) ? (int16_t)acc : 0;
    }

    /* Layer 3: h2[64] @ W3[64×64] + b3[64] -> ReLU -> h3[64] */
    for (j = 0; j < CTRL_HIDDEN; j++) {
        acc = (int32_t)ctrl->b3[j] << 6;
        for (i = 0; i < CTRL_HIDDEN; i++) {
            acc += (int32_t)h2[i] * (int32_t)ctrl->W3[i][j];
        }
        acc >>= 6;
        h3[j] = (acc > 0) ? (int16_t)acc : 0;
    }

    /* Output: h3[64] @ Wo[64×6] + bo[6] -> scores[6] (no activation) */
    for (j = 0; j < CTRL_OUTPUT; j++) {
        acc = (int32_t)ctrl->bo[j] << 6;
        for (i = 0; i < CTRL_HIDDEN; i++) {
            acc += (int32_t)h3[i] * (int32_t)ctrl->Wo[i][j];
        }
        scores[j] = (int16_t)(acc >> 6);
    }
}

uint8_t instnct_ctrl_decide(
    const InstnctController *ctrl,
    const InstnctCtrlInput *input
) {
    int16_t scores[CTRL_OUTPUT];
    int16_t best_score;
    uint8_t best_op = 0;
    int j;

    instnct_ctrl_scores(ctrl, input, scores);

    /* Argmax: highest score wins */
    best_score = scores[0];
    for (j = 1; j < CTRL_OUTPUT; j++) {
        if (scores[j] > best_score) {
            best_score = scores[j];
            best_op = (uint8_t)j;
        }
    }

    return best_op;
}
