import keras
from keras import ops


class EinsumDense(keras.layers.EinsumDense):
    """EinsumDense subclass that keeps LoRA weights in float32.

    When using mixed precision (bfloat16/float16), the parent class creates
    LoRA weights in the layer's dtype. Adam's second-moment estimate can
    underflow in low precision, producing near-zero updates. This subclass
    re-creates the LoRA variables in float32 after the parent initialises
    them, and casts them back to compute_dtype inside ``call()``.
    """

    def enable_lora(
        self, rank, a_initializer="he_uniform", b_initializer="zeros"
    ):
        super().enable_lora(rank, a_initializer, b_initializer)

        # Determine the dtype string regardless of whether .dtype is a
        # string or an object with a .name attribute.
        lora_a_dtype = self.lora_kernel_a.dtype
        if hasattr(lora_a_dtype, "name"):
            lora_a_dtype = lora_a_dtype.name

        if lora_a_dtype in ("float16", "bfloat16"):
            self._tracker.unlock()

            if self.lora_kernel_a in self._variables:
                self._variables.remove(self.lora_kernel_a)
            if self.lora_kernel_b in self._variables:
                self._variables.remove(self.lora_kernel_b)
            if self.lora_kernel_a in self._trainable_variables:
                self._trainable_variables.remove(self.lora_kernel_a)
            if self.lora_kernel_b in self._trainable_variables:
                self._trainable_variables.remove(self.lora_kernel_b)

            self.lora_kernel_a = self.add_weight(
                name="lora_kernel_a",
                shape=self.lora_kernel_a.shape,
                initializer=a_initializer,
                regularizer=self.kernel_regularizer,
                trainable=True,
                dtype="float32",
            )
            self.lora_kernel_b = self.add_weight(
                name="lora_kernel_b",
                shape=self.lora_kernel_b.shape,
                initializer=b_initializer,
                regularizer=self.kernel_regularizer,
                trainable=True,
                dtype="float32",
            )
            self._tracker.lock()

    def call(self, inputs):
        if getattr(self, "lora_rank", None):
            original_lora_a = self.lora_kernel_a
            original_lora_b = self.lora_kernel_b

            # Cast float32 LoRA weights to compute_dtype for the matmul.
            self.__dict__["lora_kernel_a"] = ops.cast(
                original_lora_a, self.compute_dtype
            )
            self.__dict__["lora_kernel_b"] = ops.cast(
                original_lora_b, self.compute_dtype
            )

            try:
                return super().call(inputs)
            finally:
                # Restore the original float32 variables so the optimiser
                # accumulates in full precision.
                self.__dict__["lora_kernel_a"] = original_lora_a
                self.__dict__["lora_kernel_b"] = original_lora_b

        return super().call(inputs)
