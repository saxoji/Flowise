# Repository Guidelines

## Project Structure & Module Organization

Flowise is a pnpm/Turbo monorepo. Core packages live in `packages/`: `server` is the Node backend and CLI, `ui` is the React/Vite frontend, `components` contains nodes, credentials, and integrations, and `agentflow`, `observe`, and `api-documentation` hold feature packages and docs. Shared media and deployment support live in `assets/`, `images/`, `docker/`, `metrics/`, and localized docs in `i18n/`. Build outputs such as `dist/`, `build/`, and coverage folders should not be edited directly.

## Build, Test, and Development Commands

- `pnpm install`: install workspace dependencies. Use Node `^20` and pnpm `^10.26.0`.
- `pnpm build`: run `turbo run build` across packages.
- `pnpm dev`: run package dev tasks in parallel; configure `.env` files in `packages/ui` and `packages/server` first.
- `pnpm start`: start the built Flowise server via `packages/server/bin`.
- `pnpm test` / `pnpm test:coverage`: run all Jest tests through Turbo.
- `pnpm lint`, `pnpm lint-fix`, `pnpm format`: run ESLint and Prettier.
- `pnpm --filter flowise-components test`: run a single package test target; also use `@flowiseai/agentflow` or `./packages/server`.

## Coding Style & Naming Conventions

TypeScript is the default for backend and integration code; React code uses JSX/TSX in `packages/ui`. Follow the root Prettier config: 4-space indentation, single quotes, no semicolons, print width 140, and no trailing commas. ESLint enforces React, hooks, accessibility, unused imports, and Prettier rules. Use `PascalCase` for classes/components/types, `camelCase` for functions and variables, and `SCREAMING_SNAKE_CASE` for environment variables.

## Testing Guidelines

Jest is used for unit tests across packages; Cypress e2e commands live in `packages/server`. Keep tests co-located with source files, for example `Foo.ts` and `Foo.test.ts` in the same directory. For new behavior, add focused tests in the touched package and run both the package test and any relevant root command before opening a PR.

## Commit & Pull Request Guidelines

Git history uses concise, Conventional-style subjects such as `fix: ...`, `chore: ...`, and scoped variants like `Fix (Agentflow) ...`. Use `feature/<name>` or `bugfix/<name>` branches. PRs should describe the change, link issues, list test commands run, include screenshots for UI changes, and call out migrations, env vars, or changesets when applicable.

## Security & Configuration Tips

Do not commit `.env` files or secrets. In credential definitions under `packages/components/credentials/`, secret values must use `type: 'password'` or `type: 'url'`, not `type: 'string'`, so the server can redact them correctly.
