# 当前项目的 Git Commit 规范

你在后续所有代码提交、生成 commit message、建议分支名时，都必须严格遵守以下规则。

## 1. Commit Message 格式
<type>(<scope>): <subject>

- type 和 subject 必填，scope 可选但推荐填写。
- 整条信息使用简体中文，subject 不以句号结尾。

## 2. type 白名单（必须从中选择）
- feat: 新增功能
- fix: 修复 Bug
- docs: 仅文档更改
- style: 格式调整（不影响代码逻辑）
- refactor: 代码重构
- perf: 性能优化
- test: 添加或修改测试
- chore: 构建/依赖/工具变动
- revert: 回退提交
- ci: CI/CD 配置变更

不在白名单中的前缀（如 update、add、change）一律禁止使用。

## 3. scope 建议
填写改动的模块名，例如：
- pic：截图模块
- match：匹配算法
- gui：界面
- calibrate：阈值校准
- deps：依赖管理

## 4. subject 要求
简明扼要地说明“做了什么”，动词开头，不要写“修改了xx”这种模糊描述。

## 5. 破坏性变更
若提交包含破坏性变更，在 type 后加 `!`，例如：
feat!: 移除旧的 SSIM 匹配逻辑
或在正文中注明：
BREAKING CHANGE: 旧的坐标常量已废弃

## 6. 分支命名
推荐使用 `feat/xxx`、`fix/xxx`、`refactor/xxx` 等前缀，例如：
- feat/match-template
- fix/marked-race-condition
- refactor/pic-module

## 7. 提交频率与内容
每完成一个独立改动就提交一次，不要把所有改动堆在一个 commit 里。
Commit message 要能独立说明这次提交的目的。

## 8. 示例
feat(match): 使用 matchTemplate 替代 SSIM
fix(marked): 修复乱序碎片下模板被抢占的问题
refactor(pic): 将截图逻辑改为窗口相对坐标并缩放至基准分辨率
chore(deps): 添加 mss 和 pygetwindow 依赖
docs: 补充阈值校准说明

## 9. 你的行为约束
- 在你需要生成 commit message 或建议提交信息时，必须严格按上述格式输出。
- 如果你在重构过程中需要分步提交，请按逻辑拆分 commit，并为每个 commit 生成符合规范的 message。
- 不要自行发明新的 type 或格式。