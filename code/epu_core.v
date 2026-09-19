module epu_core #(
    parameter integer D = 4,
    parameter integer K = 4,
    parameter integer W = 16,
    parameter integer FRAC = 8,
    parameter integer BETA_NUM = 6,
    parameter integer BETA_DEN = 1,
    parameter integer ALPHA_NUM = 3,
    parameter integer ALPHA_DEN = 5,
    parameter integer CLIP_MIN_INT = -128,
    parameter integer CLIP_MAX_INT = 127,
    parameter integer THRESHOLD_NUM = 1,
    parameter integer THRESHOLD_DEN = 4
)(
    input  wire                         clk,
    input  wire                         rst,
    input  wire                         step_en,
    input  wire signed [D*W-1:0]        z_in_flat,
    output reg  signed [D*W-1:0]        z_out_flat,
    output reg                          converged
);

    // Arithmetic contract: signed Q(W-FRAC-1).FRAC values, round-to-nearest
    // with ties away from zero, Python-equivalent shifted-score
    // normalization, alpha = ALPHA_NUM / ALPHA_DEN, and L1 convergence.
    localparam integer SCALE = (1 << FRAC);
    localparam integer CLIP_MIN_Q = CLIP_MIN_INT * SCALE;
    localparam integer CLIP_MAX_Q = CLIP_MAX_INT * SCALE;
    localparam integer THRESHOLD_Q = (SCALE * THRESHOLD_NUM) / THRESHOLD_DEN;

    integer state [0:D-1];
    integer memory_q [0:K-1][0:D-1];
    integer score_q [0:K-1];
    integer shifted_q [0:K-1];
    integer recon_q [0:D-1];
    integer next_state [0:D-1];

    integer min_score;
    integer sum_shifted;
    integer dot_acc;
    integer recon_num;
    integer next_calc;
    integer delta_q;
    integer i;
    integer j;
    reg converged_comb;

    function integer round_div_signed;
        input integer numerator;
        input integer denominator;
        begin
            if (denominator <= 0) begin
                round_div_signed = 0;
            end else if (numerator >= 0) begin
                round_div_signed = (numerator + (denominator / 2)) / denominator;
            end else begin
                round_div_signed = -(((-numerator) + (denominator / 2)) / denominator);
            end
        end
    endfunction

    function integer saturate_q;
        input integer value;
        begin
            if (value < CLIP_MIN_Q)
                saturate_q = CLIP_MIN_Q;
            else if (value > CLIP_MAX_Q)
                saturate_q = CLIP_MAX_Q;
            else
                saturate_q = value;
        end
    endfunction

    initial begin
        if (D != 4 || K != 4) begin
            $error("epu_core retained parity memory bank requires D=4 and K=4");
        end
        if (W > 31 || FRAC < 2 || FRAC >= W - 1) begin
            $error("unsupported fixed-point format");
        end
        if (BETA_DEN <= 0 || ALPHA_DEN <= 0 || THRESHOLD_DEN <= 0) begin
            $error("fixed-point denominators must be positive");
        end

        memory_q[0][0] =  2 * SCALE; memory_q[0][1] =  1 * SCALE;
        memory_q[0][2] =  0 * SCALE; memory_q[0][3] =  1 * SCALE;
        memory_q[1][0] = -1 * SCALE; memory_q[1][1] =  2 * SCALE;
        memory_q[1][2] =  1 * SCALE; memory_q[1][3] =  0 * SCALE;
        memory_q[2][0] =  1 * SCALE; memory_q[2][1] = -2 * SCALE;
        memory_q[2][2] =  2 * SCALE; memory_q[2][3] =  1 * SCALE;
        memory_q[3][0] =  0 * SCALE; memory_q[3][1] =  1 * SCALE;
        memory_q[3][2] = -1 * SCALE; memory_q[3][3] =  2 * SCALE;
    end

    always @* begin
        for (i = 0; i < D; i = i + 1) begin
            state[i] = $signed(z_in_flat[i*W +: W]);
        end

        // Python: score_i = beta * dot(z, memory_i).
        for (i = 0; i < K; i = i + 1) begin
            dot_acc = 0;
            for (j = 0; j < D; j = j + 1) begin
                dot_acc = dot_acc + state[j] * memory_q[i][j];
            end
            score_q[i] = round_div_signed(BETA_NUM * dot_acc, BETA_DEN * SCALE);
        end

        min_score = score_q[0];
        for (i = 1; i < K; i = i + 1) begin
            if (score_q[i] < min_score)
                min_score = score_q[i];
        end

        // Python: shifted_i = score_i - min(score) + 1; weights normalize
        // by sum(shifted).  Compute reconstruction directly as the same
        // weighted ratio to avoid an extra weight-quantization stage.
        sum_shifted = 0;
        for (i = 0; i < K; i = i + 1) begin
            shifted_q[i] = score_q[i] - min_score + SCALE;
            sum_shifted = sum_shifted + shifted_q[i];
        end

        delta_q = 0;
        for (j = 0; j < D; j = j + 1) begin
            recon_num = 0;
            for (i = 0; i < K; i = i + 1) begin
                recon_num = recon_num + shifted_q[i] * memory_q[i][j];
            end
            recon_q[j] = round_div_signed(recon_num, sum_shifted);
            next_calc = round_div_signed(
                ALPHA_NUM * state[j] + (ALPHA_DEN - ALPHA_NUM) * recon_q[j],
                ALPHA_DEN
            );
            next_state[j] = saturate_q(next_calc);

            if (next_state[j] >= state[j])
                delta_q = delta_q + (next_state[j] - state[j]);
            else
                delta_q = delta_q + (state[j] - next_state[j]);
        end

        converged_comb = (delta_q <= THRESHOLD_Q);
    end

    always @(posedge clk) begin
        if (rst) begin
            z_out_flat <= {D*W{1'b0}};
            converged <= 1'b0;
        end else if (step_en) begin
            for (i = 0; i < D; i = i + 1) begin
                z_out_flat[i*W +: W] <= next_state[i][W-1:0];
            end
            converged <= converged_comb;
        end
    end
endmodule
