#include <MLX90640_API.h>
#include "miao_st7735_mlx90640.h"
#include "st7735.h"
#include "string.h"
#include "MLX90640_API.h"
#include "MLX90640_I2C_Driver.h"

#define  FPS2HZ   0x02
#define  FPS4HZ   0x03
#define  FPS8HZ   0x04
#define  FPS16HZ  0x05
#define  FPS32HZ  0x06

#define  MLX90640_ADDR 0x33
#define	 RefreshRate FPS8HZ 
#define EMMISIVITY 0.95f
#define TA_SHIFT 8 //Default shift for MLX90640 in open air

#define constrain(value, min_val, max_val) \
    ((value) < (min_val) ? (min_val) : ((value) > (max_val) ? (max_val) : (value)))
#define min(a,b)  ((a)<(b)?(a):(b))

#define max(a,b)  ((a)>(b)?(a):(b))

#define abs(x)    ((x)>0?(x):(-x))


paramsMLX90640 mlx90640;
static uint16_t eeMLX90640[832];  
int status;

// start with some initial colors
float minTemp = 20.0f;
float maxTemp = 40.0f;
float centerTemp;

char tempBuffer[10];
// variables for interpolated colors
uint8_t red, green, blue;

// variables for row/column interpolation
float intPoint, val, a, b, c, d, ii;
int x, y, i, j;

// array for the 32 x 24 measured tempValues
 float tempValues[32*24];







// turn r g b to a 16bit color
uint16_t BSP_LCD_GetColor565(uint8_t red, uint8_t green, uint8_t blue){
	return ((red & 0xF8) << 8) | ((green & 0xFC) << 3) | ((blue) >> 3);
}

	  
// pass in value and figure out R G B
 
static uint16_t TempToColor(float val){
  red = constrain(255.0f / (c - b) * val - ((b * 255.0f) / (c - b)), 0, 255);

  if ((val > minTemp) & (val < a)) {
    green = constrain(255.0f / (a - minTemp) * val - (255.0f * minTemp) / (a - minTemp), 0, 255);
  }
  else if ((val >= a) & (val <= c)) {
    green = 255;
  }
  else if (val > c) {
    green = constrain(255.0f / (c - d) * val - (d * 255.0f) / (c - d), 0, 255);
  }
  else if ((val > d) | (val < a)) {
    green = 0;
  }

  if (val <= b) {
    blue = constrain(255.0f / (a - b) * val - (255.0f * b) / (a - b), 0, 255);
  }
  else if ((val > b) & (val <= d)) {
    blue = 0;
  }
  else if (val > d) {
    blue = constrain(240.0f / (maxTemp - d) * val - (d * 240.0f) / (maxTemp - d), 0, 240);
  }

  // use the displays color mapping function to get 5-6-5 color palet (R=5 bits, G=6 bits, B-5 bits)
  return BSP_LCD_GetColor565(red, green, blue);
}



/*set tempscale*/
static void setTempScale(void) {
  minTemp = 255;
  maxTemp = 0;

  for (i = 0; i < 768; i++) {
    minTemp = min(minTemp, tempValues[i]);
    maxTemp = max(maxTemp, tempValues[i]);

  }

  setAbcd();
//  drawLegend();
}




// Function to get the cutoff points in the temp vs RGB graph.
static void setAbcd(void) {
  a = minTemp + (maxTemp - minTemp) * 0.2f;
  b = minTemp + (maxTemp - minTemp) * 0.3f;
  c = minTemp + (maxTemp - minTemp) * 0.4f;
  d = minTemp + (maxTemp - minTemp) * 0.8f;
}


static void drawLegend(void) {
    uint8_t legend_height = 5;               // 图例高度（像素）
    uint8_t legend_y = 0;                     // 图例顶部位置（y=0）
    uint8_t legend_width = 120;               // 图例宽度
    uint8_t start_x = 4;                      // 图例左侧起始位置（x=4）


    float temp_range = maxTemp - minTemp;
    for (uint8_t x = 0; x < legend_width; x++) {
        float temp = minTemp + temp_range * (x / (float)legend_width);
        uint16_t color = TempToColor(temp);
        ST7735_DrawLine(start_x + x, legend_y, start_x + x, legend_y + legend_height, color);
    }

    // 显示最小/最大温度标签

    sprintf(tempBuffer, "<%2.1f       ", minTemp);
    ST7735_WriteString(2, 115, tempBuffer, Font_7x10, ST7735_WHITE, ST7735_BLACK);


    sprintf(tempBuffer, "%2.1f>       ", maxTemp);
    ST7735_WriteString(90, 115, tempBuffer, Font_7x10, ST7735_WHITE, ST7735_BLACK);
}
/*fill rectangle*/

/***********************************************      插值与绘图     ********************************************************************/
// 双线性插值
void interpolateTemperature(float *src, float *dst, uint8_t src_w, uint8_t src_h, uint8_t dst_w, uint8_t dst_h) {
    float x_ratio = (float)(src_w-1) / dst_w;
    float y_ratio = (float)(src_h-1) / dst_h;
    for (int y = 0; y < dst_h; y++) {
        for (int x = 0; x < dst_w; x++) {
            float x_src = x * x_ratio;
            float y_src = y * y_ratio;
            int x0 = (int)x_src, y0 = (int)y_src;
            float x_diff = x_src - x0;
            float y_diff = y_src - y0;
            
            float val = src[y0*src_w + x0] * (1-x_diff)*(1-y_diff)
                      + src[y0*src_w + x0+1] * x_diff*(1-y_diff)
                      + src[(y0+1)*src_w + x0] * (1-x_diff)*y_diff
                      + src[(y0+1)*src_w + x0+1] * x_diff*y_diff;
            dst[y*dst_w + x] = val;
        }
    }
}


static void drawPicture(void) {
    uint8_t cell_size = 4; // 格子大小 4x4
    uint8_t start_x = 0;   // 水平居中（128 - 32*4 = 0）
    uint8_t start_y = 16;  // 垂直居中（(128 - 24*4)/2 = 16）

    for (y = 0; y < 24; y++) {
        for (x = 0; x < 32; x++) {
            ST7735_FillRectangle(
                start_x + x * cell_size,
                start_y + (23 - y) * cell_size,
                cell_size,
                cell_size,
                TempToColor(tempValues[(31- x) + (y * 32)])
            );
        }
    }
}

static void drawPicture1(void) {
    uint8_t cell_size = 4;
    uint8_t start_x = 0;
    uint8_t start_y = 16;
    
    float dst_temp[32 * 24]; //结果数组
    
    // 对tempValues进行插值
    interpolateTemperature(tempValues, dst_temp, 32, 24, 32, 24);
    
    for (int y = 0; y < 24; y++) {
        for (int x = 0; x < 32; x++) {
            ST7735_FillRectangle(
                start_x + x * cell_size,
                start_y + (23 - y) * cell_size,
                cell_size,
                cell_size,
                TempToColor(dst_temp[(31 - x) + (y * 32)])
            );
        }
    }
}

//最临近插值

void nearestNeighborInterpolate(float *src, float *dst, uint8_t src_w, uint8_t src_h, uint8_t dst_w, uint8_t dst_h) {
    float x_ratio = (float)src_w / dst_w;
    float y_ratio = (float)src_h / dst_h;
    
    for (int y = 0; y < dst_h; y++) {
        for (int x = 0; x < dst_w; x++) {
            int src_x = (int)(x * x_ratio);
            int src_y = (int)(y * y_ratio);            
            // 防止越界
            if (src_x >= src_w) src_x = src_w - 1;
            if (src_y >= src_h) src_y = src_h - 1;
            
            dst[y * dst_w + x] = src[src_y * src_w + src_x];
        }
    }
}

static void drawPicture2(void) {
    uint8_t cell_size = 4;
    uint8_t start_x = 0;
    uint8_t start_y = 16;
    
    float dst_temp[32 * 24]; // 插值后的目标数据
    
    // 使用最近邻插值
    nearestNeighborInterpolate(tempValues, dst_temp, 32, 24, 32, 24);
    
    for (int y = 0; y < 24; y++) {
        for (int x = 0; x < 32; x++) {
            ST7735_FillRectangle(
                start_x + x * cell_size,
                start_y + (23 - y) * cell_size,
                cell_size,
                cell_size,
                TempToColor(dst_temp[(31 - x) + (y * 32)])
            );
        }
    }
}

/***********************************************      插值与绘图     ********************************************************************/



/***********************************************      边缘提取     ********************************************************************/

// Sobel 算子卷积核
const int sobel_x[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

const int sobel_y[3][3] = {
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
};


void detectEdges(float *input, uint8_t *output, uint8_t width, uint8_t height) {
    for (int y = 1; y < height - 1; y++) {       
        for (int x = 1; x < width - 1; x++) {
            float gx = 0, gy = 0;          
            // 3x3 邻域卷积
            for (int j = -1; j <= 1; j++) {
                for (int i = -1; i <= 1; i++) {
                    float pixel = input[(y + j) * width + (x + i)];
                    gx += pixel * sobel_x[j + 1][i + 1];
                    gy += pixel * sobel_y[j + 1][i + 1];
                }
            }
            uint8_t edge_strength = (uint8_t)(fabs(gx) + fabs(gy));
            edge_strength = (edge_strength > 255) ? 255 : edge_strength;
            output[y * width + x] = edge_strength;
        }
    }
}


static void drawPicture3(void) {
    uint8_t cell_size = 4;
    uint8_t start_x = 0;
    uint8_t start_y = 16;
    
    float interpolated_temp[32 * 24];  
    uint8_t edge_map[32 * 24];         
    
    //双线性插值
    interpolateTemperature(tempValues, interpolated_temp, 32, 24, 32, 24);
    //边缘检测
    detectEdges(interpolated_temp, edge_map, 32, 24);  
    //只绘制边缘（白色），其余区域黑色
    for (int y = 0; y < 24; y++) {
        for (int x = 0; x < 32; x++) {
            uint16_t color = ST7735_BLACK;  // 默认黑色
            // 如果边缘强度足够高，则显示白色
            if ((edge_map[y * 32 + x] > 10  ) && (edge_map[y * 32 + x] < 40  )){  // 阈值可调
                color = ST7735_WHITE;
            }
            ST7735_FillRectangle(
                start_x + x * cell_size,
                start_y + (23 - y) * cell_size,
                cell_size,
                cell_size,
                color
            );
        }
    }
}


/***********************************************      边缘提取     ********************************************************************/
// Read pixel data from MLX90640.
static void readTempValues(void) {

  for (uint8_t x = 0 ; x < 2 ; x++) // Read both subpages
  {
    uint16_t mlx90640Frame[834];
    status = MLX90640_GetFrameData(MLX90640_ADDR, mlx90640Frame);
    if (status < 0)
    {
//       printf("GetFrame Error: %d\r\n",status);
    }

    float vdd = MLX90640_GetVdd(mlx90640Frame, &mlx90640);
    float Ta = MLX90640_GetTa(mlx90640Frame, &mlx90640);

    float tr = Ta - TA_SHIFT; //Reflected temperature based on the sensor ambient temperature
	
	
    MLX90640_CalculateTo(mlx90640Frame, &mlx90640, EMMISIVITY, tr, tempValues);
  }
}


void mlx90640_display_init(void){
    
	printf("initialize start... \r\n");
	
	status = MLX90640_SetRefreshRate(MLX90640_ADDR, RefreshRate);
	if (status != 0) printf("\r\nset refresh erro code:%d\r\n",status);
	else printf("\r\nset refresh ok\r\n");
	
	status = MLX90640_SetChessMode(MLX90640_ADDR);
	if (status != 0) printf("\r\nset chess erro code:%d\r\n",status);
	else printf("\r\nset chess ok\r\n");
	
	
	status = MLX90640_DumpEE(MLX90640_ADDR, eeMLX90640);
	if (status != 0) printf("\r\nload system parameters error with code:%d\r\n",status);
	else printf("\r\nload system parameters ok\r\n");
  
	status = MLX90640_ExtractParameters(eeMLX90640, &mlx90640);
    if (status != 0) printf("\r\nParameter extraction failed with error code:%d\r\n",status);
    else printf("\r\nParameter extraction ok\r\n");
	
	
	printf("\r\ninitialize ok \r\n");
}

// print color to uart
void mlx90640_print_color_16bit(void){

	uint16_t color_value =0x0;

    for (int y = 0; y < 24; y++) {
        for (int x = 0; x < 32; x++) {

            //color_value =  TempToColor(tempValues[(31- x) + (y * 32)]);
			//printf("0x%04x, ",color_value);
			
			//color_value =  TempToColor(tempValues[(31- x) + (y * 32)]);
			printf("%.2f, ",tempValues[(31- x) + (y * 32)]);
			
        }
		printf("\n");
    }
	
	
}

void mlx90640_display_process(void){
	readTempValues();
	setTempScale();
	
	drawPicture1();
	drawLegend();
}
