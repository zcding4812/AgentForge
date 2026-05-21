/** 默认分页大小（Agent 列表、模型列表、会话消息等通用场景） */
export const DEFAULT_PAGE_SIZE = 10;

/** 默认生成 max_tokens */
export const DEFAULT_MAX_TOKENS = 4096;

/** max_tokens 输入下界 */
export const MAX_TOKENS_MIN = 512;

/** max_tokens 输入上界 */
export const MAX_TOKENS_MAX = 2_000_000;

/** 描述/备注类 TextArea 统一 maxLength */
export const DESCRIPTION_MAX_LENGTH = 512;

/** fetchAll 类分页循环安全上限 */
export const MAX_PAGINATION_PAGES = 50;
