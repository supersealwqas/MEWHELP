# markdown-it 浏览器构建

- 版本：`15.0.3`
- 上游项目：https://github.com/markdown-it/markdown-it
- 官方发布包：https://registry.npmjs.org/markdown-it/-/markdown-it-15.0.3.tgz
- 包内来源：`package/dist/browser/markdown-it.esm.min.mjs`
- 本地文件：`markdown-it.mjs`，仅重命名，保留官方构建的原始字节。
- 许可证：`markdown-it.LICENSE`，原样取自发布包的 `package/LICENSE`。
- 获取日期：2026-10-22。

该文件是包含依赖的浏览器 ESM 构建，运行时无需访问 CDN，也无需 npm 安装或构建。
聊天专用的配置、链接限制和图片替代文字逻辑位于上一级的 `markdown.mjs`，不修改第三方文件。
官方文件保留了 source map 注释；调试映射文件未随应用分发，不影响正常运行。

下载发布包后已核对 npm 官方元数据中的 SHA-512 完整性值：

```text
sha512-+Az7Ojz3iRGu8KYMvJ6xVhvrxP3eE+vEtJX75Ow88XBpcQNV/0lGX5ZynIIBftREo4DC2jsveAMDaeltmm8QvQ==
```

本地 `markdown-it.mjs` 的 SHA-256：

```text
d5432491790141da046b1197353705db03264f40b5727f3b4b591f5aba193e30
```
