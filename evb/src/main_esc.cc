// #include "tflite.h"
#include <stdint.h>
#include "escCntrlClass.h"
#include "am_util_stdio.h"
#include "ns_peripherals_button.h"
#include "ns_peripherals_power.h"
#include "ns_ambiqsuite_harness.h"
#include "ns_audio.h"
#include "ambiq_nnsp_const.h"
#include "ns_timer.h"
#include "ns_energy_monitor.h"
#include "nn_speech.h"
#include "downsample.h"
#include "ns_rpc_generic_data.h"
DOWNSAMPLE_CLASS downsample_inst;
int16_t buf_downsample[160];

// #define ENERGY_MEASUREMENT
#define NUM_CHANNELS 1
int volatile g_intButtonPressed = 0;

///Button Peripheral Config Struct
// #ifdef DEF_GUI_ENABLE
// ns_button_config_t button_config_nnsp = {
//     .button_0_enable = false,
//     .button_1_enable = false,
//     .button_0_flag = NULL,
//     .button_1_flag = NULL
// };
// #else

ns_button_config_t button_config_nnsp = {
    .api = &ns_button_V1_0_0,
    .button_0_enable = true,
    .button_1_enable = false,
    .button_0_flag = &g_intButtonPressed,
    .button_1_flag = NULL
};

/// Set by app when it wants to start recording, used by callback
bool volatile static g_audioRecording = false;
/// Set by callback when audio buffer has been copied, cleared by
/// app when the buffer has been consumed.
bool volatile static g_audioReady = false;
/// Audio buffer for application
int16_t static g_in16AudioDataBuffer[LEN_STFT_HOP << 1];
// uint32_t static audadcSampleBuffer[(LEN_STFT_HOP << 1) + 3];
alignas(16) uint32_t static dmaBuffer[(LEN_STFT_HOP << 1)]; // DMA target

#ifndef USE_PDM_MICROPHONE
am_hal_audadc_sample_t static workingBuffer[LEN_STFT_HOP]; 
#endif // USE_PDM_MICROPHONE

#if !defined(NS_AMBIQSUITE_VERSION_R4_1_0) && defined(NS_AUDADC_PRESENT)
am_hal_offset_cal_coeffs_array_t sOffsetCalib;
#endif

size_t ucHeapSize = NS_RPC_MALLOC_SIZE_IN_K * 4 *1024;
uint8_t ucHeap[NS_RPC_MALLOC_SIZE_IN_K * 4 *1024] __attribute__((aligned(4)));

// USB bufffers declared locally
#define MY_USB_RX_BUFSIZE 2048
#define MY_USB_TX_BUFSIZE 2048
static uint8_t my_cdc_rx_ff_buf[MY_USB_RX_BUFSIZE];
static uint8_t my_cdc_tx_ff_buf[MY_USB_TX_BUFSIZE];

// Block sent to PC
static char msg_store[30] = "Audio16bPCM_to_WAV";
static dataBlock pcmBlock = { // the block for pcm
    .length = (LEN_STFT_HOP << 1) * sizeof(int16_t),
    .dType = uint8_e,
    .description = msg_store,
    .cmd = write_cmd,
    .buffer = {.data = (uint8_t *) g_in16AudioDataBuffer, // point this to audio buffer
               .dataLength = (LEN_STFT_HOP << 1) * sizeof(int16_t)}};

// Block sent to PC for computation
char msg_compute[30] = "CalculateMFCC_Please";
dataBlock computeBlock = {  // this block is useless here actually
    .length = (LEN_STFT_HOP << 1) * sizeof(int16_t),
    .dType = uint8_e,
    .description = msg_compute,
    .cmd = extract_cmd,
    .buffer = {.data = (uint8_t *) g_in16AudioDataBuffer, // point this to audio buffer
               .dataLength = (LEN_STFT_HOP << 1) * sizeof(int16_t)}};

dataBlock IsRecordBlock;
// Block sent to PC for computation

static ns_rpc_config_t rpcConfig = {
    .api = &ns_rpc_gdo_V1_0_0,
    .mode = NS_RPC_GENERICDATA_CLIENT,
    .rx_buf = my_cdc_rx_ff_buf,
    .rx_bufLength = MY_USB_RX_BUFSIZE,
    .tx_buf = my_cdc_tx_ff_buf,
    .tx_bufLength = MY_USB_TX_BUFSIZE,
    .sendBlockToEVB_cb = NULL,
    .fetchBlockFromEVB_cb = NULL,
    .computeOnEVB_cb = NULL};
/**
* 
* @brief Audio Callback (user-defined, executes in IRQ context)
* 
* When the 'g_audioRecording' flag is set, copy the latest sample to a buffer
* and set a 'ready' flag. If recording flag isn't set, discard buffer.
* If 'ready' flag is still set, the last buffer hasn't been consumed yet,
* print a debug message and overwrite.
* 
*/
void
audio_frame_callback(ns_audio_config_t *config, uint16_t bytesCollected) {
    if (g_audioRecording) {
        // if (g_audioReady)
        //     ns_lp_printf("Warning - audio buffer wasnt consumed in time\n");
        ns_audio_getPCM_v2(config, g_in16AudioDataBuffer);
        g_audioReady = true;
    }
}

/**
 * @brief NeuralSPOT Audio config struct
 * 
 * Populate this struct before calling ns_audio_config()
 * 
 */

static ns_audio_config_t audio_config = {
    .api = &ns_audio_V2_0_0,
    .eAudioApiMode = NS_AUDIO_API_CALLBACK,
    .callback = audio_frame_callback,
    .audioBuffer = (void *)&g_in16AudioDataBuffer,

#ifdef USE_PDM_MICROPHONE
    .eAudioSource = NS_AUDIO_SOURCE_PDM,
#else
    .eAudioSource = NS_AUDIO_SOURCE_AUDADC,
#endif
    .sampleBuffer = dmaBuffer,
#if !defined(AUDIO_LEGACY) && defined(NS_AUDADC_PRESENT) && !defined(USE_PDM_MICROPHONE)
    .workingBuffer = workingBuffer,
#endif
    .numChannels = NUM_CHANNELS,
    .numSamples = LEN_STFT_HOP,
    .sampleRate = SAMPLING_RATE,
    .audioSystemHandle = NULL, // filled in by audio_init()
    .bufferHandle = NULL,
#if !defined(NS_AMBIQSUITE_VERSION_R4_1_0) && defined(NS_AUDADC_PRESENT)
    .sOffsetCalib = &sOffsetCalib,
#endif
};

// const ns_power_config_t ns_lp_audio = {
//         .eAIPowerMode           = NS_MAXIMUM_PERF,
//         .bNeedAudAdc            = true,
//         .bNeedSharedSRAM        = false,
//         .bNeedCrypto            = false,
//         .bNeedBluetooth         = false,
//         .bNeedUSB               = false,
//         .bNeedIOM               = false,
//         .bNeedAlternativeUART   = false,
//         .b128kTCM               = false,
//         .bEnableTempCo          = false,
//         .bNeedITM               = false};                                  

int main(void) {
    ns_core_config_t ns_core_cfg = {.api = &ns_core_V1_0_0};
    escCntrlClass cntrl_inst;
    int16_t *esc_output = g_in16AudioDataBuffer + LEN_STFT_HOP;
    NNSPClass *pt_nnsp;
    int16_t detected;
    int32_t tmp;
    g_audioRecording = false;
    
    NS_TRY(ns_core_init(&ns_core_cfg), "Core init failed.\n");
    // ns_power_config(&ns_lp_audio);

    ns_power_config(&ns_audio_default);

    #ifdef ENERGY_MEASUREMENT
        // ns_uart_printf_enable(); // use uart to print, uses less power
        ns_itm_printf_enable(); 
        ns_init_power_monitor_state();
        ns_set_power_monitor_state(NS_IDLE);
    #else
        ns_itm_printf_enable();
    #endif

    ns_audio_init(&audio_config);
    ns_peripheral_button_init(&button_config_nnsp);

    // initialize neural nets controller
    DOWNSAMPLE_CLASS_init(&downsample_inst);
    escCntrlClass_init(&cntrl_inst);
    pt_nnsp = (NNSPClass*) cntrl_inst.pt_nnsp;
#ifdef DEF_ACC32BIT_OPT
    ns_lp_printf("Note: You are using \"32bit\" accumulator.\n");
#else
    ns_lp_printf("Note: You are using \"64bit\" accumulator.\n");
#endif

    NS_TRY(ns_rpc_genericDataOperations_init(&rpcConfig), "RPC Init Failed\n"); // init RPC and USB
    ns_lp_printf("Before continuing, please start the PC-side application according to the following instructions\n");
    ns_lp_printf("\t$ python ../python/tools/audioview_esc.py --tty=/dev/tty.usbmodem1234561 # MacOS \n");
    ns_lp_printf("\t\tor\n");
    ns_lp_printf("\t$ python ../python/tools/audioview_esc.py --tty=/dev/serial/by-id/usb-TinyUSB_TinyUSB_Device_123456-if00  # Ubuntu \n");
    ns_lp_printf("\t\tor\n");
    ns_lp_printf("\t> python ../python/tools/audioview_esc.py --tty=COM4 # Windows \n");
    ns_lp_printf("Once the application is started, press EVB Button 0 to connect.\n");
    while (g_intButtonPressed == 0) {
        ns_delay_us(1000);
    }
    g_intButtonPressed = 0;
    ns_lp_printf("\nPress \'record\' on GUI to start, and then \'stop\' on GUI to stop recording.\n");
    ns_lp_printf("(You might change the \"--tty\" option based on your OS.)\n\n");
    ns_lp_printf("After \'stop\', check the raw recorded speech \'audio_raw.wav\' and enhanced speech \'audio_se.wav\'\n");
    ns_lp_printf("under the folder \'nnsp/evb/audio_result/\'\n\n");

    // tflite_init();
    // test_tflite();

    while (1) 
    {
        g_audioRecording = false;
        
        ns_deep_sleep();

        while (1)
        {
            // waiting for the start recording from GUI
            ns_rpc_data_computeOnPC(&computeBlock, &IsRecordBlock);
            if (IsRecordBlock.buffer.data[0]==1)
            {
                escCntrlClass_reset(&cntrl_inst);
                ns_rpc_data_clientDoneWithBlockFromPC(&IsRecordBlock);
                break;
            }
            ns_rpc_data_clientDoneWithBlockFromPC(&IsRecordBlock);
            am_hal_delay_us(20000); 
        }

        if ( !g_audioRecording )
        {
            ns_lp_printf("\nYou'd pressed the button. Program start!\n");
            g_audioRecording = true;
            am_hal_delay_us(10);   
            while (1)
            {   
                ns_set_power_monitor_state(NS_DATA_COLLECTION);
                ns_deep_sleep();
                if (g_audioReady) 
                {
                    ns_lp_printf(".");
                    // execution of each time frame data
                    pt_nnsp->pt_params->pre_gain_q1 = IsRecordBlock.buffer.data[1];
                    for (int i = 0; i < LEN_STFT_HOP; i++)
                    {
                        tmp = (int32_t) g_in16AudioDataBuffer[i] * (int32_t) pt_nnsp->pt_params->pre_gain_q1;
                        g_in16AudioDataBuffer[i] = (int16_t) (tmp >> 1); // Q1
                    }   
                    DOWNSAMPLE_CLASS_exec(
                        &downsample_inst,
                        buf_downsample,
                        g_in16AudioDataBuffer,
                        LEN_STFT_HOP);

                    detected = escCntrlClass_exec(
                        &cntrl_inst,
                        buf_downsample,
                        esc_output);
                    for (int i = 0; i < LEN_STFT_HOP; i++)
                    {
                        esc_output[i] = detected << 11;
                    }
                    esc_output[0] = detected;
                    // if (detected)
                    //     escCntrlClass_reset(&cntrl_inst);

                    ns_rpc_data_sendBlockToPC(&pcmBlock);
                    ns_rpc_data_computeOnPC(&computeBlock, &IsRecordBlock);
                    if (IsRecordBlock.buffer.data[0]==0)
                    {
                        g_audioRecording = false;
                        g_audioReady = false;
                        ns_rpc_data_clientDoneWithBlockFromPC(&IsRecordBlock);
                        break;
                    }
                    ns_rpc_data_clientDoneWithBlockFromPC(&IsRecordBlock);

                    g_audioReady = false;
                }
                
            }  // while(1)
            ns_lp_printf("\nPress button to start!\n");
        }
    } // while(1)
}
