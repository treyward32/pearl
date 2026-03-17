import axios from 'axios';

interface RpcConfig {
  rpcHost: string;
  rpcPort: number;
  rpcUser: string;
  rpcPassword: string;
}

class RpcClient {
  constructor(private config: RpcConfig) { }

  async call<Result>(method: string, params: unknown[] = []): Promise<Result> {
    const rpcData = {
      jsonrpc: '2.0',
      method,
      params,
      id: Date.now(),
    };

    const rpcUrl = `${this.config.rpcHost}:${this.config.rpcPort}`;
    try {
      const response = await axios.post(rpcUrl, rpcData, {
        auth: {
          username: this.config.rpcUser,
          password: this.config.rpcPassword,
        },
        headers: { 'Content-Type': 'application/json' },
        timeout: 10000,
      });

      if (response.data.error) {
        throw new Error(response.data.error.message || 'RPC call failed');
      }

      return response.data.result as Result;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        if (error.code === 'ECONNREFUSED') {
          throw new Error(`Cannot connect to RPC server: ${rpcUrl}`);
        }
        throw new Error(`RPC call failed: ${rpcUrl} ${error.message}`);
      }
      throw error as Error;
    }
  }
}

export { RpcClient };
export type { RpcConfig };
