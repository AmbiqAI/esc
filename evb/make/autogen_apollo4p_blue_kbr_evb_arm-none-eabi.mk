# Autogenerated file - created by NeuralSPOT make nest
INCLUDES += neuralspot/ns-core/includes-api neuralspot/ns-harness/includes-api neuralspot/ns-peripherals/includes-api neuralspot/ns-peripherals/includes-api/apollo4 neuralspot/ns-ipc/includes-api neuralspot/ns-audio/includes-api neuralspot/ns-audio/apollo4/includes-api neuralspot/ns-utils/includes-api neuralspot/ns-utils/includes-api/apollo4 neuralspot/ns-i2c/includes-api neuralspot/ns-nnsp/includes-api neuralspot/ns-usb/includes-api neuralspot/ns-usb/includes-api neuralspot/ns-rpc/includes-api neuralspot/ns-ble/includes-api extern/AmbiqSuite/R4.4.1/boards/apollo4p_blue_kbr_evb/bsp extern/AmbiqSuite/R4.4.1/CMSIS/ARM/Include extern/AmbiqSuite/R4.4.1/CMSIS/AmbiqMicro/Include extern/AmbiqSuite/R4.4.1/devices extern/AmbiqSuite/R4.4.1/mcu/apollo4p extern/AmbiqSuite/R4.4.1/mcu/apollo4p/hal/mcu extern/AmbiqSuite/R4.4.1/utils extern/AmbiqSuite/R4.4.1/third_party/FreeRTOSv10.5.1/Source/include extern/AmbiqSuite/R4.4.1/third_party/FreeRTOSv10.5.1/Source/portable/GCC/AMapollo4 extern/AmbiqSuite/R4.4.1/third_party/tinyusb/src extern/AmbiqSuite/R4.4.1/third_party/tinyusb/source/src extern/AmbiqSuite/R4.4.1/third_party/tinyusb/source/src/common extern/AmbiqSuite/R4.4.1/third_party/tinyusb/source/src/osal extern/AmbiqSuite/R4.4.1/third_party/tinyusb/source/src/class/cdc extern/AmbiqSuite/R4.4.1/third_party/tinyusb/source/src/device extern/CMSIS/CMSIS-DSP-1.15.0/Include extern/CMSIS/CMSIS-DSP-1.15.0/PrivateInclude extern/tensorflow/0264234_Nov_15_2023/. extern/tensorflow/0264234_Nov_15_2023/third_party extern/tensorflow/0264234_Nov_15_2023/third_party/flatbuffers/include extern/tensorflow/0264234_Nov_15_2023/third_party/gemmlowp extern/SEGGER_RTT/R7.70a/RTT extern/SEGGER_RTT/R7.70a/Config extern/codecs/opus-precomp/includes-api extern/AmbiqSuite/R4.4.1/third_party/cordio/ble-profiles/include/app extern/AmbiqSuite/R4.4.1/third_party/cordio/ble-profiles/sources/apps/ extern/AmbiqSuite/R4.4.1/third_party/cordio/ble-profiles/sources/apps/app/ extern/AmbiqSuite/R4.4.1/third_party/cordio/ble-profiles/sources/profiles/include extern/AmbiqSuite/R4.4.1/third_party/cordio/ble-profiles/sources/profiles extern/AmbiqSuite/R4.4.1/third_party/cordio/ble-profiles/sources/profiles/gatt extern/AmbiqSuite/R4.4.1/third_party/cordio/ble-profiles/sources/services extern/AmbiqSuite/R4.4.1/third_party/cordio/wsf/include extern/AmbiqSuite/R4.4.1/third_party/cordio/wsf/sources/port/freertos extern/AmbiqSuite/R4.4.1/third_party/cordio/wsf/sources extern/AmbiqSuite/R4.4.1/third_party/cordio/wsf/sources/util extern/AmbiqSuite/R4.4.1/third_party/cordio/ble-host/include extern/AmbiqSuite/R4.4.1/third_party/cordio/ble-host/sources/stack/cfg extern/AmbiqSuite/R4.4.1/third_party/cordio/ble-host/sources/hci/ambiq extern/AmbiqSuite/R4.4.1/third_party/cordio/ble-host/sources/hci/ambiq/cooper extern/AmbiqSuite/R4.4.1/third_party/cordio/ble-host/sources/stack/hci/ extern/AmbiqSuite/R4.4.1/third_party/cordio/ble-host/sources/stack/smp/ extern/AmbiqSuite/R4.4.1/third_party/cordio/ble-host/sources/stack/dm/ extern/AmbiqSuite/R4.4.1/third_party/cordio/ble-host/sources/stack/l2c/ extern/AmbiqSuite/R4.4.1/third_party/cordio/ble-host/sources/stack/att/ extern/AmbiqSuite/R4.4.1/third_party/cordio/ble-host/sources/sec/common/ extern/AmbiqSuite/R4.4.1/third_party/cordio/uecc extern/AmbiqSuite/R4.4.1/third_party/cordio/devices extern/erpc/R1.9.1/includes-api
libraries += libs/apollo4p_blue_kbr_evb/arm-none-eabi/neuralspot/ns-core/ns-core.a libs/apollo4p_blue_kbr_evb/arm-none-eabi/neuralspot/ns-harness/ns-harness.a libs/apollo4p_blue_kbr_evb/arm-none-eabi/neuralspot/ns-peripherals/ns-peripherals.a libs/apollo4p_blue_kbr_evb/arm-none-eabi/neuralspot/ns-ipc/ns-ipc.a libs/apollo4p_blue_kbr_evb/arm-none-eabi/neuralspot/ns-audio/ns-audio.a libs/apollo4p_blue_kbr_evb/arm-none-eabi/neuralspot/ns-utils/ns-utils.a libs/apollo4p_blue_kbr_evb/arm-none-eabi/neuralspot/ns-i2c/ns-i2c.a libs/apollo4p_blue_kbr_evb/arm-none-eabi/neuralspot/ns-nnsp/ns-nnsp.a libs/apollo4p_blue_kbr_evb/arm-none-eabi/neuralspot/ns-usb/ns-usb.a libs/apollo4p_blue_kbr_evb/arm-none-eabi/neuralspot/ns-rpc/ns-rpc.a libs/apollo4p_blue_kbr_evb/arm-none-eabi/neuralspot/ns-ble/ns-ble.a libs/apollo4p_blue_kbr_evb/arm-none-eabi/extern/AmbiqSuite/R4.4.1/ambiqsuite.a libs/apollo4p_blue_kbr_evb/arm-none-eabi/extern/SEGGER_RTT/R7.70a/segger_rtt.a libs/apollo4p_blue_kbr_evb/arm-none-eabi/extern/codecs/opus-precomp/codecs.a libs/apollo4p_blue_kbr_evb/arm-none-eabi/extern/erpc/R1.9.1/erpc.a libs/apollo4p_blue_kbr_evb/arm-none-eabi/neuralspot/ns-usb/ns-usb-overrides.a libs/extern/AmbiqSuite/R4.4.1/lib/apollo4p/libam_hal.a libs/extern/AmbiqSuite/R4.4.1/lib/apollo4p/blue_kbr_evb/libam_bsp.a libs/extern/CMSIS/CMSIS-DSP-1.15.0/lib/libCMSISDSP-m4-gcc.a libs/extern/tensorflow/0264234_Nov_15_2023/lib/libtensorflow-microlite-cm4-gcc-release.a libs/extern/codecs/opus-precomp/libs/libopus.a libs/extern/AmbiqSuite/R4.4.1/third_party/cordio/lib/apollo4p/blue_kbr_evb/gcc/cordio.a
override_libraries += libs/apollo4p_blue_kbr_evb/arm-none-eabi/neuralspot/ns-usb/ns-usb-overrides.a
local_app_name := basic_tf_stub
