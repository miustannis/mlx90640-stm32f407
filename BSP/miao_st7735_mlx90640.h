#include <stdio.h>
#include <math.h>
#include "main.h"

void mlx90640_display_init(void);
void mlx90640_display_process(void);


static uint16_t TempToColor(float val);
static void setTempScale(void);
static void setAbcd(void);
static void drawLegend(void);
static void drawMeasurement(void);
static void drawPicture(void);
static void readTempValues(void);
void interpolateTemperature(float *src, float *dst, uint8_t src_w, uint8_t src_h, uint8_t dst_w, uint8_t dst_h);
void mlx90640_print_color_16bit(void);
