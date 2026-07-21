export interface DistilURLOptions {
  /** Proxy host. Default "127.0.0.1". */
  host?: string;
  /** Proxy port. Default 8788 (or DISTIL_PORT). */
  port?: number;
}

/** Base URL of a local distil proxy (honors DISTIL_BASE_URL). */
export function distilBaseURL(opts?: DistilURLOptions): string;

/** distil proxy base URL with the OpenAI `/v1` segment appended. */
export function distilOpenAIBaseURL(opts?: DistilURLOptions): string;

export const DEFAULT_PORT: number;
export const DEFAULT_HOST: string;
