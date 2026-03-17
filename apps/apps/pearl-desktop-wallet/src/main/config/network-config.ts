/**
 * Central network configuration for the wallet
 * All network-specific settings should be consumed from here
 */

import { MAINNET_DEFAULT_PEER_ADDRESSES, TESTNET_DEFAULT_PEER_ADDRESSES } from "./consts";

export type Network = 'mainnet' | 'testnet';

export interface NetworkConfig {
    name: Network;
    displayName: string;
    rpcPort: number;
    walletFlag: string;
    dataSubdir: string;
    addressPrefix: string;
    defaultPeerAddress: string;
    defaultPeerPort: number;
}

const nodeIndex = Math.floor(Math.random() * 3);

const mainnetDefaultPeerAddress = MAINNET_DEFAULT_PEER_ADDRESSES[nodeIndex];
const testnetDefaultPeerAddress = TESTNET_DEFAULT_PEER_ADDRESSES[nodeIndex];

const NETWORK_CONFIGS: Record<Network, NetworkConfig> = {
    mainnet: {
        name: 'mainnet',
        displayName: 'Mainnet',
        rpcPort: 8335,
        walletFlag: '',  // No flag for mainnet (default)
        dataSubdir: 'mainnet',
        addressPrefix: 'prl1',
        defaultPeerAddress: mainnetDefaultPeerAddress,
        defaultPeerPort: 44108,
    },
    testnet: {
        name: 'testnet',
        displayName: 'Testnet',
        rpcPort: 8335,
        walletFlag: '--testnet2',
        dataSubdir: 'testnet2',
        addressPrefix: 'tprl1',
        defaultPeerAddress: testnetDefaultPeerAddress,
        defaultPeerPort: 44112,
    },
};

let currentNetwork: Network = 'mainnet';

export function getCurrentNetwork(): Network {
    return currentNetwork;
}

export function setCurrentNetwork(network: Network): void {
    currentNetwork = network;
}

export function getCurrentNetworkConfig(): NetworkConfig {
    return NETWORK_CONFIGS[currentNetwork];
}

export function getAllNetworks(): Network[] {
    return Object.keys(NETWORK_CONFIGS) as Network[];
}
