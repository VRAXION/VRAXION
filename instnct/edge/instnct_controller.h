/*
 * INSTNCT Edge — ReLU mutation controller
 * ========================================
 * 10->64->64->64->6 ReLU network decides which mutation op to use.
 * Pure integer version: weights quantized to int8, activations int16.
 * ~9.4K params, <1ms per decision on Arduino.
 */

#ifndef INSTNCT_CONTROLLER_H
#define INSTNCT_CONTROLLER_H

#include <stdint.h>

#define CTRL_INPUT  10
#define CTRL_HIDDEN 64
#define CTRL_OUTPUT  6

/* Quantized weights: int8 (scaled by 128) */
typedef struct {
    int8_t W1[CTRL_INPUT][CTRL_HIDDEN];
    int8_t b1[CTRL_HIDDEN];
    int8_t W2[CTRL_HIDDEN][CTRL_HIDDEN];
    int8_t b2[CTRL_HIDDEN];
    int8_t W3[CTRL_HIDDEN][CTRL_HIDDEN];
    int8_t b3[CTRL_HIDDEN];
    int8_t Wo[CTRL_HIDDEN][CTRL_OUTPUT];
    int8_t bo[CTRL_OUTPUT];
} InstnctController;

/* Controller input features (all scaled to [0, 100]) */
typedef struct {
    uint8_t accuracy;       /* current eval accuracy × 100 */
    uint8_t pain;           /* 100 if last rejected, decays */
    uint8_t reward;         /* 100 if last accepted, decays */
    uint8_t rate_add;       /* rolling 100-step accept % for add */
    uint8_t rate_flip;      /* rolling accept % for flip */
    uint8_t rate_theta;     /* rolling accept % for theta */
    uint8_t rate_channel;   /* rolling accept % for channel */
    uint8_t rate_reverse;   /* rolling accept % for reverse */
    uint8_t rate_remove;    /* rolling accept % for remove */
    uint8_t progress;       /* training progress 0-100 */
} InstnctCtrlInput;

/*
 * Forward pass: pick the best mutation op.
 *
 * ctrl:    controller weights (from flash/ROM)
 * input:   current training state features
 * Returns: op index [0-5] = {add, flip, theta, channel, reverse, remove}
 */
uint8_t instnct_ctrl_decide(
    const InstnctController *ctrl,
    const InstnctCtrlInput *input
);

/*
 * Forward pass with confidence: returns all 6 scores.
 * scores[6] filled with relative confidence per op (higher = more confident).
 */
void instnct_ctrl_scores(
    const InstnctController *ctrl,
    const InstnctCtrlInput *input,
    int16_t scores[CTRL_OUTPUT]
);

#endif /* INSTNCT_CONTROLLER_H */
