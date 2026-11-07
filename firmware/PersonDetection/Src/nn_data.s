/* The two model parts (NN_MODEL_A/B, set by the Makefile: make MODEL=<name>) and the test image, included as
   binary files into internal flash (paths relative to firmware/). TFLM wants the models 16-byte aligned. */

    .section .rodata.nn_model_a,"a",%progbits
    .balign 16
    .global nn_model_a
nn_model_a:
    .incbin NN_MODEL_A

    .section .rodata.nn_model_b,"a",%progbits
    .balign 16
    .global nn_model_b
nn_model_b:
    .incbin NN_MODEL_B

#ifdef NN_MODEL_A2
    .section .rodata.nn_model_a2,"a",%progbits
    .balign 16
    .global nn_model_a2
nn_model_a2:
    .incbin NN_MODEL_A2
#endif

    .section .rodata.nn_test_image,"a",%progbits
    .balign 4
    .global nn_test_image
nn_test_image:
#ifdef NN_TEST_IMAGE
    .incbin NN_TEST_IMAGE
#else
    .incbin "PersonDetection/test/skier_192x192.rgb"
#endif
