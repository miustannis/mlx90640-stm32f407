/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "dma.h"
#include "i2c.h"
#include "spi.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "st7735.h"
#include "testimg.h"
#include "MLX90640_API.h"
#include "MLX90640_I2C_Driver.h"
#include "miao_st7735_mlx90640.h"
#include "model.h"
#include "micro_temp_net_weights.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define INPUT_MEAN 27.1f
#define INPUT_STD  1.86f
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
char send[20] = {0}; //传输用字符数组，储存字符串

uint8_t rx_buffer = 0; //接收字符缓冲区

char rx_receiver[20] = {0}; //接收用字符数组

uint8_t rx_num = 0; //接收字符数量

uint8_t  uart_start_flag = 0; //接收标志位

__IO  uint32_t uart_gettick = 0; //系统时钟比较变量

uint16_t print_counter = 0;

MicroTempNet CNNmodel;

extern float tempValues[32*24];

char tempbuffer[3] = {0};  

float input[1 * 32 * 24] = { 0 };
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */
void UART_RX_PROC(void);

void CNN_PROC(void);

void z_score_normalize(float* input, int size);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_SPI1_Init();
  MX_USART1_UART_Init();
  MX_I2C2_Init();
  /* USER CODE BEGIN 2 */
  ST7735_Init();
  ST7735_FillScreen(ST7735_BLACK);
  //MLX90640_I2CInit();
  //ST7735_Init();
  mlx90640_display_init();
  
  HAL_UART_Receive_IT(&huart1 , &rx_buffer,  1);
//  ST7735_FillScreen(ST7735_CYAN);
 // initialization
  micro_temp_net_init(&CNNmodel, 
					conv1_weight, conv1_bias,
					conv2_weight, conv2_bias,
					fc1_weight, fc2_bias,
					fc2_weight, fc2_bias);
  //ST7735_DrawImage8bit_Img2LCD(0,0,128,128,(uint8_t*)gImage_tower);


  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
	 mlx90640_display_process();
	  
	 //mlx90640_print_color_16bit();
	  //UART_RX_PROC();
	  
	  CNN_PROC();
	 //HAL_Delay(100);
	  	// ST7735_FillScreen(ST7735_BLACK);
   //mlx90640_print_color_16bit();
	  
   //printf("test ok \r\n");
	  
   //HAL_Delay(500);
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 168;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */
void UART_RX_PROC(void)
{
    if((( uwTick- uart_gettick ) >=10) && (( uwTick- uart_gettick ) <=600) && (uart_start_flag == 1)){
    if(rx_receiver[0] == '&' ){
		if(print_counter < 50){
	      mlx90640_print_color_16bit();
		  print_counter++;
		}
		else{
		printf("test over \r\n");
		}

		
  }
    //printf("receiver check \r\n");
    rx_num = 0;
    uart_start_flag = 0;
}

}

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
  if(rx_num == 0){
  uart_gettick = uwTick; 
  uart_start_flag = 1;  
  }
  if(uart_start_flag == 1){
  rx_receiver[rx_num] = rx_buffer;
  rx_num++;  
  }
  HAL_UART_Receive_IT(&huart1 , &rx_buffer,  1);
}


void CNN_PROC(void){


	//input
    for (int y = 0; y < 24; y++) {
        for (int x = 0; x < 32; x++){
			input[y * 32 + x] =  tempValues[(31- x) + (y * 32)]; 
        }
    }
	z_score_normalize(input,768);
	
    // airun
    float output[NUM_CLASSES];
    micro_temp_net_predict(&CNNmodel, input, output);

    // postproc
    int predicted_class = 0;
    float max_prob = output[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (output[i] > max_prob) {
            max_prob = output[i];
            predicted_class = i;
        }
    }
	predicted_class += 1;
	sprintf(tempbuffer, "%d " , predicted_class);
	ST7735_WriteString(65, 115, tempbuffer, Font_7x10, ST7735_WHITE, ST7735_BLACK);

}


void z_score_normalize(float* input, int size) {
    for (int i = 0; i < size; i++) {
        input[i] = (input[i] - INPUT_MEAN) / INPUT_STD;
    }
}
/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
