INCLUDES += neuralspot/ns-nnsp/includes-api
libraries += libs/ns-nnsp.a

ACC32BIT_OPT:=0		# 1: accumulator 32bit, 0: acc 64bit
NNSP_MODE:=0        # 1: run vad+kws+s2i, 0: s2i only
DEF_USE_PDM_MICROPHONE:=0 # 1: use pdm microphone, 0: use i2s microphone
