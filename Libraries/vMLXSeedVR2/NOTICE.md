# SeedVR2 implementation provenance

The initial transformer/VAE modules, latent creator, scheduler, quantization helper,
and color transfer were adapted from xocialize/seedvr2-mlx-swift, commit
1dfa161ba9522bdf1295015fd0b00209a4e30093:
https://github.com/xocialize/seedvr2-mlx-swift

The upstream README declares the port code MIT, and each copied source file carries
SPDX-License-Identifier: MIT. Attribution: MVS Collective (xocialize) and contributors.
The implementation references filipstrand/mflux and ByteDance Seed's SeedVR2.
Model weights have their own Apache-2.0 license and are not included here.

MIT License

Copyright (c) MVS Collective (xocialize) and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
