package com.nextransit.mrzbench.photo.ocr

data class OcrResult(
    val engineName: String,
    val line1: String,
    val line2: String,
    val conf1: Int,
    val conf2: Int,
    val ms: Long,
    val error: String? = null,
    /** True when a second checksum-valid reading was almost as likely as
     * the returned one.  A wrong-but-checksum-valid line is
     * indistinguishable from a correct one for the caller, so the
     * ambiguity has to travel with the result. */
    val ambiguous: Boolean = false,
    /** Log-probability gap between the best and the runner-up reading,
     * in thousandths (0 when there is no runner-up). */
    val altMargin: Int = 0,
    /** The runner-up reading, empty when there is none. */
    val altLine2: String = "",
    /**
     * line 2 的字段在结构上是否可能（mrz_ocr.h `mrz_ocr_line2_struct_ok`）：
     * 两个日期都是真实日历日、出生不在未来、到期不早于出生。
     *
     * 与五个校验位**无关**——校验位只能证明 44 个字符自洽，不能证明它们是
     * 文字。assets 语料上 28 张校验通过的里 19 张国籍码不存在、1 张日期不存在。
     *
     * 默认 true 是为了不影响非 native 引擎（Paddle/tesseract/mlkit）：
     * 它们不产出这个信号，不应该因此被判为不可信。
     */
    val structOk: Boolean = true,
    /**
     * 国籍字段（line2 10..12）可疑（`mrz_ocr_line2_nat_suspect`）：
     * 三字母、不含 '<'、且不是 ISO-3166-1 alpha-3 或 ICAO 保留码。
     *
     * 只作告警：德国印 `D<<`（含 '<' 故不告警）、科索沃印 `RKS`（会误报）。
     * 含 '<' 的字段一律放行。
     */
    val natSuspect: Boolean = false,
    /**
     * 签发国（line1 2..4）与国籍（line2 10..12）不一致（`mrz_ocr_line1_country_mismatch`）。
     *
     * 这是本轮唯一能抓住 **line1 单字符错误**的信号：41 张真机照片上它标出
     * 4 张、4 张全错，而置信度与 ambiguous 合起来一个都没抓到（有一张 line1
     * 错了但置信度高达 98）。
     *
     * 只作告警：难民旅行证可以合法地不一致，字段含 `<`（德国 `D<<`）时一律放行。
     */
    val countryMismatch: Boolean = false,
    /**
     * 签发国码（line1 2..4）可疑（`mrz_ocr_line1_issuer_suspect`）：
     * 三字母、不含 '<'、且不是 ISO-3166-1 alpha-3 或 ICAO 保留码。
     *
     * 这是 [natSuspect] 在 line1 上的镜像，而**补上它才是对的**：line2 有五个
     * 校验位，line1 一个都没有——恰恰是没有校验位的那一行最需要内容检查，
     * 之前却完全没人查。
     *
     * 41 张真机照片上 line1 读错的 4 张，签发国码全是伪造的（`HUG`/`CHY`/
     * `CNN`/`ATA`），没有任何一个国家会签发；而两套语料的全部真值里，
     * 签发国码都是合法 alpha-3。
     *
     * 只作告警：含 '<' 的字段一律放行（德国印 `D<<`）。
     */
    val issuerSuspect: Boolean = false,
) {
    fun isEmpty(): Boolean = line1.isEmpty() && line2.isEmpty()
    val ok: Boolean get() = error == null && (line1.isNotEmpty() || line2.isNotEmpty())

    /**
     * 可以展示 / 落库的结果：识别成功，且校验位之外的否决位没有触发。
     *
     * 这是 [ok] 的严格版本。`ok` 只说明「拿到了两行非空文本」；一个校验位
     * 被凑出来的噪声行同样满足它。调用方要展示给用户或写入数据库时应该用
     * 这个。
     */
    val trustworthy: Boolean get() = ok && structOk && !natSuspect && !issuerSuspect
}
