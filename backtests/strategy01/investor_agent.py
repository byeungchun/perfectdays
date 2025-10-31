"""Investor agent encapsulating buy and sell decision logic for simulations."""

from __future__ import annotations

from collections import defaultdict


class InvestorAgent:
    """Handles buy/sell decisions and tracks portfolio state for each ticker."""

    def __init__(
        self,
        *,
        budget: float,
        z_vol: float,
        vwap_rel_std_max: float,
        enhanced_sell: bool,
        sell_after_days: int = 5,
    ) -> None:
        self.initial_budget = budget
        self.cash_balance = budget
        self.z_vol = z_vol
        self.vwap_rel_std_max = vwap_rel_std_max
        self.enhanced_sell = enhanced_sell
        self.sell_after_days = sell_after_days
        self.shares_owned: dict[str, list] = defaultdict(list)
        self.revenue_records: dict[str, list] = defaultdict(list)
        self.simulation_results: dict[str, list] = defaultdict(list)

    def process_stock(
        self,
        simuldate,
        prepared,
    ) -> None:
        (
            ticker,
            _ts_hist,
            tstarget,
            vwap_mean,
            vwap_std,
            vol_mean,
            vol_std,
            vol,
        ) = prepared

        simul_res = {
            "simuldate": simuldate,
            "ticker": ticker,
            "popularity": None,
            "vwap_stability": None,
            "investment_signal": None,
            "invest_amount": None,
            "shares_bought": None,
            "invest_flag": None,
        }

        holdings = self.shares_owned[ticker]
        prev_results = self.simulation_results[ticker]

        self.cash_balance = self._sell(
            holdings,
            simuldate,
            tstarget,
            ticker,
            simul_res,
        )

        vol_pop, vwap_stable, investment_signal = self._decide_buy_signal(
            vwap_mean,
            vwap_std,
            vol_mean,
            vol_std,
            vol,
        )
        simul_res["popularity"] = vol_pop
        simul_res["vwap_stability"] = vwap_stable
        simul_res["investment_signal"] = investment_signal

        self.cash_balance = self._buy(
            simul_res,
            prev_results,
            tstarget,
            ticker,
            simuldate,
        )

        self.simulation_results[ticker].append(simul_res)

    def results(self) -> tuple[dict, dict, dict, float]:
        return self.simulation_results, self.shares_owned, self.revenue_records, self.cash_balance

    def _decide_buy_signal(
        self,
        vwap_mean: float,
        vwap_std: float,
        vol_mean: float,
        vol_std: float,
        vol: float,
        *,
        min_abs_volume: float | None = None,
        vwap_abs_std_max: float | None = None,
        require_both: bool = True,
        rel_eps: float = 1e-12,
    ) -> tuple[bool, bool, bool]:
        if vol_std and vol_std > 0:
            vol_z = (vol - vol_mean) / vol_std
            vol_pop_z = vol_z >= self.z_vol
        else:
            vol_pop_z = vol > vol_mean
        vol_pop_abs = min_abs_volume is not None and vol >= min_abs_volume
        vol_pop = bool(vol_pop_z or vol_pop_abs)

        if abs(vwap_mean) > rel_eps:
            vwap_stable_rel = vwap_std <= abs(vwap_mean) * self.vwap_rel_std_max
        else:
            abs_cap = vwap_abs_std_max if vwap_abs_std_max is not None else 1e-6
            vwap_stable_rel = vwap_std <= abs_cap
        vwap_stable_abs = vwap_abs_std_max is None or vwap_std <= vwap_abs_std_max
        vwap_stable = bool(vwap_stable_rel and vwap_stable_abs)

        investment_signal = (vol_pop and vwap_stable) if require_both else (vol_pop or vwap_stable)
        return bool(vol_pop), bool(vwap_stable), bool(investment_signal)

    def _sell(
        self,
        holdings: list,
        simuldate,
        tstarget,
        ticker: str,
        simul_res: dict,
        *,
        min_hold_days: int = 0,
        take_profit_pct: float | None = None,
        stop_loss_pct: float | None = None,
        trailing_stop_pct: float | None = None,
        price_field: str = "priceClose",
        partial_sell_ratio: float | None = None,
        prioritize_time_exit: bool = False,
    ) -> float:
        if not self.enhanced_sell:
            if not holdings or ("sold_date" in holdings[-1]):
                return self.cash_balance
            position = holdings[-1]
            invest_date = position["buy_date"]
            days_since_invest = (simuldate - invest_date).days
            if days_since_invest >= self.sell_after_days:
                target_price = tstarget[price_field]
                target_amount = position["shares"] * target_price
                revenue = target_amount - (position["shares"] * position["buy_price"])
                simul_res["invest_flag"] = "sold"
                self.cash_balance += target_amount
                holdings.append(
                    {
                        "ticker": ticker,
                        "shares": -position["shares"],
                        "sold_price": target_price,
                        "sold_date": simuldate,
                        "buy_date": invest_date,
                    }
                )
                self.revenue_records[ticker].append(
                    {
                        "ticker": ticker,
                        "sell_date": simuldate,
                        "revenue": int(revenue),
                    }
                )
            else:
                simul_res["invest_flag"] = "holding"
            return self.cash_balance

        open_idx = None
        for idx in range(len(holdings) - 1, -1, -1):
            record = holdings[idx]
            if record.get("shares", 0) > 0 and "buy_date" in record and "sold_date" not in record:
                open_idx = idx
                break
        if open_idx is None:
            return self.cash_balance

        position = holdings[open_idx]
        current_price = float(tstarget[price_field])
        buy_price = float(position["buy_price"])
        shares_held = int(position["shares"])
        buy_date = position["buy_date"]
        holding_days = (simuldate - buy_date).days

        max_price_since_buy = float(position.get("max_price_since_buy", buy_price))
        if current_price > max_price_since_buy:
            max_price_since_buy = current_price
        position["max_price_since_buy"] = max_price_since_buy

        profit_pct = (current_price - buy_price) / buy_price if buy_price else 0.0
        drawdown_pct = (
            (max_price_since_buy - current_price) / max_price_since_buy if max_price_since_buy > 0 else 0.0
        )

        time_exit = self.sell_after_days is not None and holding_days >= self.sell_after_days
        can_exit_by_rule = holding_days >= min_hold_days
        tp_trigger = take_profit_pct is not None and can_exit_by_rule and profit_pct >= float(take_profit_pct)
        sl_trigger = stop_loss_pct is not None and can_exit_by_rule and profit_pct <= -float(stop_loss_pct)
        ts_trigger = trailing_stop_pct is not None and can_exit_by_rule and drawdown_pct >= float(trailing_stop_pct)

        reason = None
        should_sell = False
        if prioritize_time_exit and time_exit:
            should_sell = True
            reason = "time_exit"
        else:
            if sl_trigger:
                should_sell = True
                reason = "stop_loss"
            elif ts_trigger:
                should_sell = True
                reason = "trailing_stop"
            elif tp_trigger:
                should_sell = True
                reason = "take_profit"
            elif time_exit:
                should_sell = True
                reason = "time_exit"

        if not should_sell:
            simul_res["invest_flag"] = "holding"
            return self.cash_balance

        if partial_sell_ratio is not None and 0 < partial_sell_ratio < 1:
            shares_to_sell = max(1, int(shares_held * partial_sell_ratio))
            full_close = shares_to_sell >= shares_held
        else:
            shares_to_sell = shares_held
            full_close = True

        target_amount = shares_to_sell * current_price
        revenue = target_amount - (shares_to_sell * buy_price)
        self.cash_balance += target_amount

        sell_record = {
            "ticker": ticker,
            "shares": -shares_to_sell,
            "sold_price": current_price,
            "sold_date": simuldate,
            "buy_date": buy_date,
            "sell_reason": reason,
        }
        holdings.append(sell_record)
        self.revenue_records[ticker].append(
            {
                "ticker": ticker,
                "shares_sold": shares_to_sell,
                "sold_reason": reason,
                "sold_price": current_price,
                "bought_price": buy_price,
                "sell_date": simuldate,
                "revenue": int(revenue),
            }
        )

        if not full_close:
            position["shares"] = shares_held - shares_to_sell
            simul_res["invest_flag"] = "holding"
        else:
            simul_res["invest_flag"] = "sold"
        return self.cash_balance

    def _buy(
        self,
        simul_res: dict,
        prev_results: list,
        tstarget,
        ticker: str,
        simuldate,
    ) -> float:
        if (
            simul_res["investment_signal"]
            and prev_results
            and prev_results[-1]["invest_flag"] in ["no_action", "sold"]
            and simul_res["invest_flag"] is None
        ):
            invest_amount = self.initial_budget * 0.1
            price_per_share = tstarget["priceClose"]
            shares_to_buy = int(invest_amount // price_per_share) - 1
            if shares_to_buy <= 0:
                simul_res["invest_flag"] = "no_action"
                return self.cash_balance
            total_investment = shares_to_buy * price_per_share
            simul_res["invest_amount"] = int(total_investment)
            simul_res["shares_bought"] = int(shares_to_buy)
            simul_res["invest_flag"] = "investing"
            self.cash_balance -= total_investment
            self.shares_owned[ticker].append(
                {
                    "ticker": ticker,
                    "shares": shares_to_buy,
                    "buy_price": price_per_share,
                    "buy_date": simuldate,
                }
            )
        else:
            if simul_res["invest_flag"] is None:
                simul_res["invest_flag"] = "no_action"
        return self.cash_balance
