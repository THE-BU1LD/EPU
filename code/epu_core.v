module epu_core #(
    parameter D = 4,
    parameter K = 4,
    parameter W = 8
)(
    input  wire                   clk,
    input  wire                   rst,
    input  wire                   step_en,
    input  wire signed [D*W-1:0]  z_in_flat,
    output reg  signed [D*W-1:0]  z_out_flat,
    output reg                    converged
);

    reg signed [W-1:0] state [0:D-1];
    reg signed [W-1:0] memory [0:K-1][0:D-1];
    reg signed [15:0] score [0:K-1];
    reg signed [15:0] min_score;
    reg signed [15:0] shifted [0:K-1];
    reg signed [W-1:0] recon [0:D-1];
    reg signed [W-1:0] next_state [0:D-1];
    integer i, j;

    initial begin
        memory[0][0] =  2; memory[0][1] =  1; memory[0][2] =  0; memory[0][3] =  1;
        memory[1][0] = -1; memory[1][1] =  2; memory[1][2] =  1; memory[1][3] =  0;
        memory[2][0] =  1; memory[2][1] = -2; memory[2][2] =  2; memory[2][3] =  1;
        memory[3][0] =  0; memory[3][1] =  1; memory[3][2] = -1; memory[3][3] =  2;
    end

    always @(*) begin
        for (i = 0; i < D; i = i + 1) begin
            state[i] = z_in_flat[i*W +: W];
        end
    end

    always @(*) begin
        for (i = 0; i < K; i = i + 1) begin
            score[i] = 0;
            for (j = 0; j < D; j = j + 1) begin
                score[i] = score[i] + state[j] * memory[i][j];
            end
        end
        min_score = score[0];
        for (i = 1; i < K; i = i + 1) begin
            if (score[i] < min_score) min_score = score[i];
        end
        for (i = 0; i < K; i = i + 1) begin
            shifted[i] = score[i] - min_score + 1;
        end
        for (j = 0; j < D; j = j + 1) begin
            recon[j] = 0;
            for (i = 0; i < K; i = i + 1) begin
                recon[j] = recon[j] + (shifted[i][W-1:0] * memory[i][j]) / 4;
            end
            next_state[j] = (state[j] * 6 + recon[j] * 4) / 10;
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            z_out_flat <= 0;
            converged <= 1'b0;
        end else if (step_en) begin
            for (i = 0; i < D; i = i + 1) begin
                z_out_flat[i*W +: W] <= next_state[i];
            end
            converged <= 1'b0;
        end
    end
endmodule
