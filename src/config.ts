import path from "path";
import "dotenv/config";
import os from "os";
import fs from "fs-extra";
import pino from "pino";
import { kokoroModelPrecision, whisperModels } from "./types/shorts";

const defaultLogLevel: pino.Level = "info";
const defaultPort = 3123;
const whisperVersion = "1.7.1";
const defaultWhisperModel: whisperModels = "medium.en"; // possible options: "tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v1", "large-v2", "large-v3", "large-v3-turbo"

// Create the global logger
const versionNumber = process.env.npm_package_version;
export const logger = pino({
  level: process.env.LOG_LEVEL || defaultLogLevel,
  timestamp: pino.stdTimeFunctions.isoTime,
  formatters: {
    level: (label) => {
      return { level: label };
    },
  },
  base: {
    pid: process.pid,
    version: versionNumber,
  },
});

export class Config {
  private dataDirPath: string;
  private libsDirPath: string;
  private staticDirPath: string;

  public installationSuccessfulPath: string;
  public whisperInstallPath: string;
  public videosDirPath: string;
  public tempDirPath: string;
  public packageDirPath: string;
  public musicDirPath: string;
  public pexelsApiKey: string;
  public mediaProvider: "pexels" | "pollinations" | "cloudflare";
  public pollinationsApiKey?: string;
  public pollinationsStylePrompt: string;
  public cloudflareAccountId?: string;
  public cloudflareApiToken?: string;
  public cloudflareImageModel: "flux" | "sdxl";
  public cloudflareFluxSteps: number;
  public logLevel: pino.Level;
  public whisperVerbose: boolean;
  public port: number;
  public runningInDocker: boolean;
  public devMode: boolean;
  public whisperVersion: string = whisperVersion;
  public whisperModel: whisperModels = defaultWhisperModel;
  public kokoroModelPrecision: kokoroModelPrecision = "fp32";

  // docker-specific, performance-related settings to prevent memory issues
  public concurrency?: number;
  public videoCacheSizeInBytes: number | null = null;

  constructor() {
    this.dataDirPath =
      process.env.DATA_DIR_PATH ||
      path.join(os.homedir(), ".ai-agents-az-video-generator");
    this.libsDirPath = path.join(this.dataDirPath, "libs");

    this.whisperInstallPath = path.join(this.libsDirPath, "whisper");
    this.videosDirPath = path.join(this.dataDirPath, "videos");
    this.tempDirPath = path.join(this.dataDirPath, "temp");
    this.installationSuccessfulPath = path.join(
      this.dataDirPath,
      "installation-successful",
    );

    fs.ensureDirSync(this.dataDirPath);
    fs.ensureDirSync(this.libsDirPath);
    fs.ensureDirSync(this.videosDirPath);
    fs.ensureDirSync(this.tempDirPath);

    this.packageDirPath = path.join(__dirname, "..");
    this.staticDirPath = path.join(this.packageDirPath, "static");
    this.musicDirPath = path.join(this.staticDirPath, "music");

    this.pexelsApiKey = process.env.PEXELS_API_KEY as string;
    this.mediaProvider =
      (process.env.MEDIA_PROVIDER as "pexels" | "pollinations" | "cloudflare") ||
      "pexels";
    this.pollinationsApiKey = process.env.POLLINATIONS_API_KEY || undefined;
    this.pollinationsStylePrompt =
      process.env.POLLINATIONS_STYLE_PROMPT ||
      "hand-drawn 2D doodle animation style, flat colors, bold black outlines, no photorealism, simple children's illustration style";
    this.cloudflareAccountId = process.env.CLOUDFLARE_ACCOUNT_ID || undefined;
    this.cloudflareApiToken = process.env.CLOUDFLARE_API_TOKEN || undefined;
    // "flux" (default): fast, 4 steps, no negative_prompt support.
    // "sdxl": supports negative_prompt for real anti-artifact control,
    // slower (20 steps default) and more neurons/image.
    this.cloudflareImageModel =
      (process.env.CLOUDFLARE_IMAGE_MODEL as "flux" | "sdxl") || "flux";
    // Default 4 matches flux-1-schnell's documented safe range (1-4 steps).
    // Configurable to test whether 5-8 improves anatomy consistency without
    // a code change — untested territory for this specific hosted model.
    this.cloudflareFluxSteps = process.env.CLOUDFLARE_FLUX_STEPS
      ? parseInt(process.env.CLOUDFLARE_FLUX_STEPS)
      : 4;
    this.logLevel = (process.env.LOG_LEVEL || defaultLogLevel) as pino.Level;
    this.whisperVerbose = process.env.WHISPER_VERBOSE === "true";
    this.port = process.env.PORT ? parseInt(process.env.PORT) : defaultPort;
    this.runningInDocker = process.env.DOCKER === "true";
    this.devMode = process.env.DEV === "true";

    if (process.env.WHISPER_MODEL) {
      this.whisperModel = process.env.WHISPER_MODEL as whisperModels;
    }
    if (process.env.KOKORO_MODEL_PRECISION) {
      this.kokoroModelPrecision = process.env
        .KOKORO_MODEL_PRECISION as kokoroModelPrecision;
    }

    this.concurrency = process.env.CONCURRENCY
      ? parseInt(process.env.CONCURRENCY)
      : undefined;

    if (process.env.VIDEO_CACHE_SIZE_IN_BYTES) {
      this.videoCacheSizeInBytes = parseInt(
        process.env.VIDEO_CACHE_SIZE_IN_BYTES,
      );
    }
  }

  public ensureConfig() {
    if (this.mediaProvider === "pexels" && !this.pexelsApiKey) {
      throw new Error(
        "PEXELS_API_KEY environment variable is missing. Get your free API key: https://www.pexels.com/api/key/ - see how to run the project: https://github.com/gyoridavid/short-video-maker. Alternatively, set MEDIA_PROVIDER=pollinations or MEDIA_PROVIDER=cloudflare to use free AI-generated images instead.",
      );
    }
    if (
      this.mediaProvider === "cloudflare" &&
      (!this.cloudflareAccountId || !this.cloudflareApiToken)
    ) {
      throw new Error(
        "CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN environment variables are required when MEDIA_PROVIDER=cloudflare. Create a free account at https://dash.cloudflare.com and generate a Workers AI-scoped API token.",
      );
    }
  }
}

export const KOKORO_MODEL = "onnx-community/Kokoro-82M-v1.0-ONNX";
