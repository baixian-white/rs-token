export type Channel = "none" | "awgn" | "rayleigh";
export type Protection = "none" | "ldpc";
export type Priority = "alert" | "balanced" | "detail";

export interface Sample {
  id: string;
  name: string;
  name_zh: string;
  class_id: number;
}

export interface Prediction {
  class_id: number;
  name: string;
  name_zh: string;
  score: number;
}

export interface Decision {
  k: number;
  quality: string;
  reason: string;
  source_bits: number;
  transmitted_bits: number;
  budget_limited: boolean;
  channel_limited: boolean;
  next_threshold_db: number | null;
  stabilized: boolean;
}

export interface InferenceResult {
  request: {
    filename: string;
    original_width: number;
    original_height: number;
    channel: Channel;
    snr_db: number;
    protection: Protection;
    priority: Priority;
    auto_k: boolean;
    seed: number;
  };
  decision: Decision;
  channel: {
    theoretical_ber: number;
    raw_ber: number;
    post_ber: number;
    bit_errors: number;
    index_errors: number;
    token_error_rate: number;
    error_grid: number[];
    ldpc_success: boolean | null;
    ldpc_iterations: number | null;
  };
  metrics: {
    psnr_db: number;
    lpips: number | null;
    bandwidth_saving_pct: number;
    encode_ms: number;
    channel_ms: number;
    decode_ms: number;
    total_ms: number;
  };
  task_predictions: Prediction[];
  recon_predictions: Prediction[];
  images: {
    input: string;
    reconstruction: string;
    progressive: Array<{ k: number; bits: number; image: string }>;
  };
  model: {
    name: string;
    token_grid: string;
    codebook_size: number;
    max_layers: number;
  };
}
