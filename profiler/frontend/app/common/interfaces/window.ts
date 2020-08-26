declare global {
  /** The interface for a TensorBoard Enviroments. */
  interface TensorBoardEnv {
    IN_COLAB?: boolean;
  }

  /** The interface for the browser's window. */
  interface Window {
    TENSORBOARD_ENV?: TensorBoardEnv;
    google?: any;
  }
}

export {};
